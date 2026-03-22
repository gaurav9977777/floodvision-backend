"""
MODULE 1 — Micro Flood Hotspot Engine  (v3 — Dynamic City Support)
════════════════════════════════════════════════════════════════════
Key fix in v3: The engine now uses the ACTUAL city centre passed via
set_city() instead of the hardcoded Nagpur CITY_CENTER constant.
This means hotspot polygons appear at the correct geographic location
when the frontend switches cities.

Scoring layers (unchanged from v2):
  1. Topographic Depression   0.30
  2. Rainfall Intensity        0.25
  3. Drainage Efficiency       0.20
  4. Water Proximity           0.15
  5. Soil Infiltration Proxy   0.10
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy.ndimage import uniform_filter, distance_transform_edt
from .data_generator import (
    generate_ward_boundaries, generate_mock_dem,
    generate_rainfall_grid, CITY_CENTER, WARD_SPACING, GRID_ROWS, GRID_COLS
)

GRID_H, GRID_W = 20, 20

LAYER_WEIGHTS = {
    "topographic_depression": 0.30,
    "rainfall_intensity":     0.25,
    "drainage_efficiency":    0.20,
    "water_proximity":        0.15,
    "soil_infiltration":      0.10,
}

RISK_THRESHOLDS = {"High": 0.60, "Medium": 0.35}
OUTPUT_THRESHOLD = 0.20


def _norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


class HotspotEngine:
    """
    Vectorised flood hotspot engine.
    Uses the city centre supplied via set_city() for all coordinate maths,
    so hotspot polygons appear at the correct location on the map.
    """

    def __init__(self):
        # Start with default city (Nagpur / Bhilai area)
        self._city_lat = CITY_CENTER[0]
        self._city_lon = CITY_CENTER[1]

        self.wards    = generate_ward_boundaries()
        self.dem      = generate_mock_dem()
        self.rainfall = generate_rainfall_grid()
        self._cache: Optional[Dict[str, Any]] = None
        self._drain_health_grid: Optional[np.ndarray] = None

    # ── Public API ────────────────────────────────────────────────────────

    def set_city(self, city_id: str, lat: float, lon: float) -> None:
        """
        Update the city centre used for grid → lat/lon mapping.
        Invalidates cache so next get_hotspots() recomputes with new coords.
        """
        if abs(lat - self._city_lat) > 0.001 or abs(lon - self._city_lon) > 0.001:
            self._city_lat = lat
            self._city_lon = lon
            self._cache = None           # force recompute
            self._drain_health_grid = None

    def refresh(self,
                dem: Optional[np.ndarray] = None,
                rainfall: Optional[np.ndarray] = None) -> None:
        if dem is not None:
            self.dem = dem
        if rainfall is not None:
            self.rainfall = rainfall
        self._drain_health_grid = None
        self._cache = None

    def get_hotspots(self, **kwargs) -> Dict[str, Any]:
        if self._cache is None:
            self._cache = self._build_geojson()
        return self._cache

    def get_high_risk_cells(self) -> List[Tuple[float, float]]:
        fc = self.get_hotspots()
        out = []
        for feat in fc["features"]:
            if feat["properties"]["risk_category"] == "High":
                coords = feat["geometry"]["coordinates"][0]
                lats = [c[1] for c in coords]
                lons = [c[0] for c in coords]
                out.append((sum(lats)/len(lats), sum(lons)/len(lons)))
        return out

    # ── Scoring layers ────────────────────────────────────────────────────

    def _layer_topographic_depression(self) -> np.ndarray:
        smooth = uniform_filter(self.dem.astype(np.float32), size=5)
        sink   = np.clip(smooth - self.dem, 0, None)
        return _norm(sink)

    def _layer_rainfall_intensity(self) -> np.ndarray:
        return _norm(np.clip(self.rainfall, 0, None))

    def _layer_drainage_efficiency(self) -> np.ndarray:
        if self._drain_health_grid is None:
            self._drain_health_grid = self._build_drain_grid()
        return _norm(self._drain_health_grid)

    def _layer_water_proximity(self) -> np.ndarray:
        mask = np.zeros((GRID_H, GRID_W), dtype=bool)
        mask[6:10, 2:6]   = True
        mask[12:16, 8:13] = True
        dist      = distance_transform_edt(~mask).astype(np.float32)
        proximity = 1.0 / (1.0 + dist)
        return _norm(proximity)

    def _layer_soil_infiltration(self) -> np.ndarray:
        dem_f          = self.dem.astype(np.float32)
        grad_y, grad_x = np.gradient(dem_f)
        slope          = np.sqrt(grad_x**2 + grad_y**2)
        flat           = 1.0 / (1.0 + slope)
        depression     = self._layer_topographic_depression()
        return _norm(0.5 * _norm(flat) + 0.5 * depression)

    # ── Grid helpers ──────────────────────────────────────────────────────

    def _build_drain_grid(self) -> np.ndarray:
        """Map ward flood-risk onto the 20×20 grid using the CURRENT city centre."""
        grid     = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        lat0     = self._city_lat - 0.05
        lon0     = self._city_lon - 0.0625
        cell_lat = (GRID_ROWS * WARD_SPACING) / GRID_H
        cell_lon = (GRID_COLS * WARD_SPACING) / GRID_W

        for r in range(GRID_H):
            for c in range(GRID_W):
                cell_lat_c = lat0 + (r + 0.5) * cell_lat
                cell_lon_c = lon0 + (c + 0.5) * cell_lon
                best_risk, best_dist = 0.5, 1e9
                for ward in self.wards:
                    coords = ward["geometry"]["coordinates"][0]
                    wlat   = (coords[0][1] + coords[2][1]) / 2
                    wlon   = (coords[0][0] + coords[2][0]) / 2
                    dist   = (cell_lat_c - wlat)**2 + (cell_lon_c - wlon)**2
                    if dist < best_dist:
                        best_dist = dist
                        best_risk = ward["properties"]["flood_risk_base"]
                grid[r, c] = best_risk
        return grid

    def _grid_to_latlon(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid cell to lat/lon using the CURRENT city centre."""
        lat0     = self._city_lat - 0.05
        lon0     = self._city_lon - 0.0625
        cell_lat = (GRID_ROWS * WARD_SPACING) / GRID_H
        cell_lon = (GRID_COLS * WARD_SPACING) / GRID_W
        return (lat0 + (row + 0.5) * cell_lat,
                lon0 + (col + 0.5) * cell_lon)

    @staticmethod
    def _classify(score: float) -> str:
        if score >= RISK_THRESHOLDS["High"]:
            return "High"
        elif score >= RISK_THRESHOLDS["Medium"]:
            return "Medium"
        return "Low"

    # ── Pipeline ──────────────────────────────────────────────────────────

    def _compute_composite_score(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        layers = {
            "topographic_depression": self._layer_topographic_depression(),
            "rainfall_intensity":     self._layer_rainfall_intensity(),
            "drainage_efficiency":    self._layer_drainage_efficiency(),
            "water_proximity":        self._layer_water_proximity(),
            "soil_infiltration":      self._layer_soil_infiltration(),
        }
        composite = sum(
            LAYER_WEIGHTS[name] * arr for name, arr in layers.items()
        ).astype(np.float32)
        return _norm(composite), layers

    def _build_geojson(self) -> Dict[str, Any]:
        composite, layers = self._compute_composite_score()

        lat_span = (GRID_ROWS * WARD_SPACING) / GRID_H / 2
        lon_span = (GRID_COLS * WARD_SPACING) / GRID_W / 2

        features      = []
        count_by_cat  = {"High": 0, "Medium": 0, "Low": 0}
        max_score     = 0.0

        for r in range(GRID_H):
            for c in range(GRID_W):
                score = float(composite[r, c])
                if score < OUTPUT_THRESHOLD:
                    continue

                lat, lon = self._grid_to_latlon(r, c)
                category = self._classify(score)
                count_by_cat[category] += 1
                max_score = max(max_score, score)

                layer_breakdown = {
                    name: round(float(arr[r, c]), 3)
                    for name, arr in layers.items()
                }

                features.append({
                    "type": "Feature",
                    "properties": {
                        "hotspot_id":           f"H{r:02d}{c:02d}",
                        "risk_score":           round(score, 3),
                        "risk_category":        category,
                        "estimated_depth_m":    round(score * 2.1, 2),
                        "rainfall_anomaly_mm":  round(float(self.rainfall[r, c]), 1),
                        "elevation_m":          round(float(self.dem[r, c]), 1),
                        "layer_scores":         layer_breakdown,
                        "primary_driver":       max(layer_breakdown,
                                                     key=lambda k: layer_breakdown[k]),
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [lon - lon_span, lat - lat_span],
                            [lon + lon_span, lat - lat_span],
                            [lon + lon_span, lat + lat_span],
                            [lon - lon_span, lat + lat_span],
                            [lon - lon_span, lat - lat_span],
                        ]],
                    },
                })

        return {
            "type":     "FeatureCollection",
            "features": features,
            "metadata": {
                "total_hotspots":    len(features),
                "high_risk_count":   count_by_cat["High"],
                "medium_risk_count": count_by_cat["Medium"],
                "low_risk_count":    count_by_cat["Low"],
                "max_risk_score":    round(max_score, 3),
                "city_lat":          self._city_lat,
                "city_lon":          self._city_lon,
                "layer_weights":     LAYER_WEIGHTS,
                "algorithm":         "Weighted Multi-Layer Composite v3",
            },
        }
