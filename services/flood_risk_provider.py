"""
services/flood_risk_provider.py
================================
Bridge between the ConvLSTM rainfall prediction algorithm and FloodVision-Pro.

Provides ward-level flood risk scores (0–1) derived from:
  1. ConvLSTM model output (primary signal, when model is available)
  2. OpenWeather real-time rainfall (live 7-day history)
  3. SRTM elevation susceptibility (static modifier)
  4. Drain health score (from drain_intelligence)

The fused score replaces the static `flood_risk_base` used by the old
budget optimizer, making allocations respond to actual rainfall conditions.

Integration flow:
  WardReadiness  →  FloodRiskProvider.get_ward_risk_scores()
  BudgetOptimizer → FloodRiskProvider.get_ward_risk_scores()
  DigitalTwin    → FloodRiskProvider.get_ward_risk_scores()

Fallback chain (so the system always works):
  ConvLSTM model → OpenWeather API → elevation heuristic → static baseline
"""

import os
import math
import time
import logging
import hashlib
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
OWM_API_KEY    = os.getenv("OPENWEATHER_API_KEY", "d06a46823b09e732b2cf9a09cac9cf13")
MODEL_PATH     = os.getenv("FLOOD_MODEL_PATH", "model/flood_model.h5")
MAX_RAIN_MM    = float(os.getenv("MAX_RAINFALL_MM", "300.0"))
TIME_STEPS     = int(os.getenv("MODEL_TIME_STEPS", "7"))
CACHE_TTL      = 1800   # 30 min – don't hammer OWM/model per request

# Fusion weights
W_MODEL    = 0.55   # ConvLSTM prediction
W_RAIN     = 0.25   # recent raw OWM rainfall
W_ELEV     = 0.12   # elevation susceptibility
W_DRAIN    = 0.08   # drain health (inverse)

# ── In-memory cache ───────────────────────────────────────────────────────────
_risk_cache: Dict[str, dict] = {}   # ward_id → {score, expires_at}
_model_instance = None


# ── ConvLSTM model loader ─────────────────────────────────────────────────────

def _load_model():
    global _model_instance
    if _model_instance is not None:
        return _model_instance
    try:
        import tensorflow as tf
        if os.path.exists(MODEL_PATH):
            _model_instance = tf.keras.models.load_model(MODEL_PATH)
            logger.info(f"ConvLSTM model loaded: {MODEL_PATH}")
        else:
            logger.warning(f"Model not found at {MODEL_PATH} — using OWM fallback")
    except Exception as exc:
        logger.warning(f"Model load failed: {exc} — using OWM fallback")
    return _model_instance


def _predict_with_model(lat: float, lon: float) -> Optional[float]:
    """
    Run ConvLSTM inference for a single (lat, lon) point.
    Returns risk score [0,1] or None if model unavailable.
    """
    model = _load_model()
    if model is None:
        return None
    try:
        # Build spatial grid centred on ward centroid
        H = W = 30
        grid_size = 0.025   # degrees per cell
        daily_rain = _fetch_owm_rainfall(lat, lon, days=TIME_STEPS)
        grids = []
        for rain_mm in daily_rain:
            # Gaussian spread from centre
            cy, cx = H // 2, W // 2
            sigma = 0.3 * H
            yy, xx = np.mgrid[:H, :W]
            gauss = np.exp(-0.5 * (((yy - cy) / sigma)**2 + ((xx - cx) / sigma)**2))
            gauss = gauss / gauss.max()
            grids.append((gauss * rain_mm).astype(np.float32))

        arr = np.stack(grids, axis=0)                       # (T, H, W)
        arr_norm = np.clip(arr / MAX_RAIN_MM, 0, 1)
        X = arr_norm[np.newaxis, ..., np.newaxis]           # (1, T, H, W, 1)
        score = float(np.squeeze(model.predict(X, verbose=0)))
        return float(np.clip(score, 0.0, 1.0))
    except Exception as exc:
        logger.warning(f"Model inference failed for ({lat},{lon}): {exc}")
        return None


# ── OpenWeather rainfall fetcher ──────────────────────────────────────────────

def _fetch_owm_rainfall(lat: float, lon: float, days: int = 7) -> list:
    """Fetch recent daily rainfall totals from OWM current + forecast."""
    if not OWM_API_KEY:
        return [0.0] * days
    try:
        import urllib.request, json as _json
        url = (f"https://api.openweathermap.org/data/2.5/forecast"
               f"?lat={lat}&lon={lon}&appid={OWM_API_KEY}&units=metric&cnt=40")
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = _json.loads(resp.read())

        # Aggregate 3h buckets → daily
        daily: dict = {}
        for item in data.get("list", []):
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(item["dt"], tz=timezone.utc).date()
            rain_3h = item.get("rain", {}).get("3h", 0.0)
            daily[dt] = daily.get(dt, 0.0) + float(rain_3h)

        values = list(daily.values())[:days]
        # Pad with zeros if fewer than `days` returned
        while len(values) < days:
            values.insert(0, 0.0)
        return values[-days:]
    except Exception as exc:
        logger.debug(f"OWM fetch failed: {exc}")
        return [0.0] * days


def _rainfall_risk(lat: float, lon: float) -> float:
    """
    Convert recent 7-day rainfall to a risk score [0,1].
    Uses IMD heavy-rain threshold (64.5 mm/day) as reference.
    """
    daily = _fetch_owm_rainfall(lat, lon, days=7)
    if not daily or all(v == 0 for v in daily):
        return 0.0
    max_day   = max(daily)
    total_7d  = sum(daily)
    # Score: blend peak day and 7-day total
    peak_score  = min(1.0, max_day / 115.6)    # 115.6mm = very heavy rain (IMD)
    total_score = min(1.0, total_7d / 300.0)   # 300mm/week = high cumulative
    return round(0.6 * peak_score + 0.4 * total_score, 4)


# ── Elevation susceptibility ──────────────────────────────────────────────────

def _elevation_risk(lat: float, lon: float, city_avg_elev: float = 315.0) -> float:
    """
    Approximate elevation risk from city DEM.
    Lower than city average → higher flood susceptibility.
    Uses OpenTopoData if available, otherwise falls back to heuristic.
    """
    try:
        import urllib.request, json as _json
        url = f"https://api.opentopodata.org/v1/srtm30m?locations={lat},{lon}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data  = _json.loads(resp.read())
        elev = float(data["results"][0].get("elevation") or city_avg_elev)
    except Exception:
        # Heuristic: use lat/lon seeded noise around city average
        seed = int(abs(lat * 1000 + lon * 100)) % 997
        np.random.seed(seed)
        elev = city_avg_elev + np.random.normal(0, 12)

    # Logistic: elev < avg_elev → risk approaches 1
    diff = city_avg_elev - elev   # positive = lower than average
    risk = 1 / (1 + math.exp(-diff / 8.0))
    return round(float(np.clip(risk, 0.0, 1.0)), 4)


# ── Drain health contribution ─────────────────────────────────────────────────

def _drain_risk(drain_health_score: float) -> float:
    """Convert drain health (0–100) to risk contribution [0,1]."""
    return round(float(np.clip(1.0 - drain_health_score / 100.0, 0.0, 1.0)), 4)


# ── Main fused risk scorer ────────────────────────────────────────────────────

def get_fused_risk(
    lat: float,
    lon: float,
    drain_health: float = 60.0,
    static_base: float = 0.5,
    city_avg_elev: float = 315.0,
    force_refresh: bool = False,
) -> dict:
    """
    Compute fused flood risk score for a ward centroid.

    Returns dict:
      risk_score      : float [0,1]  ← primary signal for budget optimizer
      model_score     : float | None
      rainfall_score  : float
      elevation_score : float
      drain_score     : float
      source          : str  ('convlstm' | 'owm' | 'static')
      daily_rainfall  : list[float]
    """
    cache_key = f"{round(lat,3)}:{round(lon,3)}"
    now = time.time()

    if not force_refresh and cache_key in _risk_cache:
        entry = _risk_cache[cache_key]
        if entry["expires_at"] > now:
            return entry["data"]

    # ── Component scores ──────────────────────────────────────────────────
    model_score  = _predict_with_model(lat, lon)
    rain_score   = _rainfall_risk(lat, lon)
    elev_score   = _elevation_risk(lat, lon, city_avg_elev)
    d_score      = _drain_risk(drain_health)
    daily_rain   = _fetch_owm_rainfall(lat, lon, days=7)

    # ── Fuse ─────────────────────────────────────────────────────────────
    if model_score is not None:
        fused = (W_MODEL * model_score
                 + W_RAIN  * rain_score
                 + W_ELEV  * elev_score
                 + W_DRAIN * d_score)
        source = "convlstm"
    else:
        # No model — redistribute model weight to rain & elevation
        w_rain = W_RAIN  + W_MODEL * 0.6
        w_elev = W_ELEV  + W_MODEL * 0.25
        w_base = 1.0 - w_rain - w_elev - W_DRAIN
        fused  = (w_rain  * rain_score
                  + w_elev  * elev_score
                  + W_DRAIN * d_score
                  + w_base  * static_base)
        source = "owm" if rain_score > 0 else "static"

    fused = float(np.clip(fused, 0.0, 1.0))

    result = {
        "risk_score":      round(fused, 4),
        "model_score":     round(model_score, 4) if model_score is not None else None,
        "rainfall_score":  round(rain_score, 4),
        "elevation_score": round(elev_score, 4),
        "drain_score":     round(d_score, 4),
        "source":          source,
        "daily_rainfall_mm": [round(r, 1) for r in daily_rain],
    }

    _risk_cache[cache_key] = {"data": result, "expires_at": now + CACHE_TTL}
    return result


# ── Batch scorer (all wards) ──────────────────────────────────────────────────

def get_ward_risk_scores(
    ward_map: dict,
    drain_health_map: Optional[Dict[str, float]] = None,
    city_avg_elev: float = 315.0,
) -> Dict[str, dict]:
    """
    Compute fused risk scores for all wards.

    Parameters
    ----------
    ward_map : dict  ward_id → {lat, lon, flood_risk_base, ...}
    drain_health_map : dict  ward_id → health_score (0–100)
    city_avg_elev : float

    Returns
    -------
    dict  ward_id → full risk dict (from get_fused_risk)
    """
    drain_health_map = drain_health_map or {}
    scores = {}
    for ward_id, info in ward_map.items():
        lat  = info.get("lat",  info.get("centroid_lat",  21.145))
        lon  = info.get("lon",  info.get("centroid_lon",  79.088))
        dh   = drain_health_map.get(ward_id, 60.0)
        base = info.get("flood_risk_base", 0.5)
        scores[ward_id] = get_fused_risk(
            lat, lon,
            drain_health=dh,
            static_base=base,
            city_avg_elev=city_avg_elev,
        )
    return scores
