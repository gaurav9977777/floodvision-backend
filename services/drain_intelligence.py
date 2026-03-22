"""
MODULE 2 — Last-Meter Drain Failure Intelligence
Predicts clogged / structurally weak drains using multi-factor scoring.
"""

import numpy as np
from typing import Dict, Any, List
from .data_generator import generate_ward_boundaries, generate_drain_network


class DrainIntelligence:
    """Computes Drain Health Index (0–100) for every drain segment."""

    WEIGHTS = {
        "age_penalty": 0.20,
        "cleaning_staleness": 0.30,
        "slope_discontinuity": 0.20,
        "rainfall_anomaly": 0.15,
        "drain_type": 0.15,
    }

    def __init__(self, city_lat=None, city_lon=None):
        from .data_generator import _city_center
        lat = city_lat or _city_center[0]
        lon = city_lon or _city_center[1]
        self.wards    = generate_ward_boundaries(city_lat=lat, city_lon=lon)
        self.drains   = generate_drain_network(self.wards)
        self._computed = None

    def reinit(self, city_lat: float, city_lon: float) -> None:
        """Regenerate drain data for a new city centre."""
        self.wards    = generate_ward_boundaries(city_lat=city_lat, city_lon=city_lon)
        self.drains   = generate_drain_network(self.wards)
        self._computed = None

    def _drain_type_score(self, dtype: str) -> float:
        return {"open": 0.45, "covered": 0.75, "stormwater": 0.90}.get(dtype, 0.6)

    def _cleaning_score(self, last_cleaned_days: int) -> float:
        """Exponential decay: fresh clean = 1.0, 365 days ago ≈ 0.1"""
        return float(np.exp(-last_cleaned_days / 180))

    def _age_score(self, age_years: int) -> float:
        return max(0.0, 1.0 - age_years / 40)

    def _slope_discontinuity(self, flood_risk: float) -> float:
        """Proxy: high flood risk area → likely terrain mismatch → poor drainage."""
        return 1.0 - flood_risk

    def _rainfall_resilience(self, flood_risk: float) -> float:
        return 1.0 - min(flood_risk * 1.2, 1.0)

    def _compute_health(self, drain: Dict, ward_risk: float) -> float:
        base = drain["properties"]["health_score_base"]
        # Apply modifiers from multi-factor model
        age_s = self._age_score(drain["properties"]["age_years"])
        clean_s = self._cleaning_score(drain["properties"]["last_cleaned_days"])
        slope_s = self._slope_discontinuity(ward_risk)
        rain_s = self._rainfall_resilience(ward_risk)
        dtype_s = self._drain_type_score(drain["properties"]["drain_type"])

        # Composite score weighted sum (0–1) × 100
        composite = (
            self.WEIGHTS["age_penalty"] * age_s +
            self.WEIGHTS["cleaning_staleness"] * clean_s +
            self.WEIGHTS["slope_discontinuity"] * slope_s +
            self.WEIGHTS["rainfall_anomaly"] * rain_s +
            self.WEIGHTS["drain_type"] * dtype_s
        )

        # Blend with base (data-driven) score
        final = 0.6 * base + 0.4 * composite * 100
        return round(float(np.clip(final, 2, 100)), 1)

    @staticmethod
    def _classify(score: float) -> str:
        if score >= 80:
            return "Healthy"
        elif score >= 50:
            return "Moderate"
        return "High Risk"

    def compute_drain_health(self) -> List[Dict[str, Any]]:
        # Build ward risk lookup
        ward_risk = {w["properties"]["ward_id"]: w["properties"]["flood_risk_base"]
                     for w in self.wards}

        results = []
        for drain in self.drains:
            wid = drain["properties"]["ward_id"]
            risk = ward_risk.get(wid, 0.5)
            health = self._compute_health(drain, risk)
            cat = self._classify(health)

            feat = {
                "type": "Feature",
                "properties": {
                    **drain["properties"],
                    "health_score": health,
                    "health_category": cat,
                    "maintenance_priority": 3 - ["Healthy", "Moderate", "High Risk"].index(cat),
                    "estimated_repair_cost_inr": int((100 - health) * 4200),
                },
                "geometry": drain["geometry"],
            }
            results.append(feat)
        return results

    def get_drain_health(self) -> Dict[str, Any]:
        if self._computed is None:
            self._computed = self.compute_drain_health()

        features = self._computed
        cats = [f["properties"]["health_category"] for f in features]
        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "total_drains": len(features),
                "healthy": cats.count("Healthy"),
                "moderate": cats.count("Moderate"),
                "high_risk": cats.count("High Risk"),
                "avg_health_score": round(
                    sum(f["properties"]["health_score"] for f in features) / len(features), 1
                )
            }
        }

    def get_ward_avg_health(self) -> Dict[str, float]:
        """Utility: per-ward average drain health for readiness module."""
        if self._computed is None:
            self._computed = self.compute_drain_health()
        drains = self._computed
        ward_scores: Dict[str, List[float]] = {}
        for d in drains:
            wid = d["properties"]["ward_id"]
            ward_scores.setdefault(wid, []).append(d["properties"]["health_score"])
        return {wid: round(np.mean(scores), 1) for wid, scores in ward_scores.items()}
