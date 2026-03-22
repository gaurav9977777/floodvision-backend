"""
SovereignFlood — Dynamic Data Generator
Generates realistic ward/drain/road data centered on ANY city.
Key fix: all coordinates use the requested city centre (lat, lon),
not a hardcoded Nagpur constant.
"""

import numpy as np
from typing import List, Dict, Any, Tuple

# Default fallback (Bhilai, Chhattisgarh)
DEFAULT_CENTER = (21.209, 81.378)
GRID_ROWS    = 4
GRID_COLS    = 5
WARD_SPACING = 0.025   # ~2.75 km per ward

# Module-level mutable city centre — updated via set_city_center()
_city_center: Tuple[float, float] = DEFAULT_CENTER

# Kept for backward-compat with any direct imports
CITY_CENTER = DEFAULT_CENTER


def set_city_center(lat: float, lon: float) -> None:
    """Update the city centre used by all generators."""
    global _city_center, CITY_CENTER
    _city_center = (lat, lon)
    CITY_CENTER  = (lat, lon)


def get_city_center() -> Tuple[float, float]:
    return _city_center


def generate_ward_boundaries(
    city_lat: float = None,
    city_lon: float = None,
) -> List[Dict[str, Any]]:
    """Generate 20 ward polygons centred on (city_lat, city_lon)."""
    lat_c = city_lat if city_lat is not None else _city_center[0]
    lon_c = city_lon if city_lon is not None else _city_center[1]

    lat0 = lat_c - (GRID_ROWS / 2) * WARD_SPACING
    lon0 = lon_c - (GRID_COLS / 2) * WARD_SPACING

    ward_names = [
        "Sector 1",   "Sector 2",   "Sector 3",   "Sector 4",   "Sector 5",
        "Sector 6",   "Sector 7",   "Sector 8",   "Sector 9",   "Sector 10",
        "Zone North", "Zone South", "Zone East",  "Zone West",  "Zone Central",
        "Riverside",  "Old City",   "New Colony", "Industrial", "Outskirts",
    ]
    risk_profile = [
        0.82, 0.45, 0.91, 0.33, 0.76,
        0.58, 0.40, 0.35, 0.68, 0.55,
        0.72, 0.63, 0.88, 0.79, 0.41,
        0.66, 0.84, 0.74, 0.61, 0.47,
    ]

    wards = []
    for idx in range(GRID_ROWS * GRID_COLS):
        r   = idx // GRID_COLS
        c   = idx %  GRID_COLS
        lat = lat0 + r * WARD_SPACING
        lon = lon0 + c * WARD_SPACING
        s   = WARD_SPACING
        wards.append({
            "type": "Feature",
            "properties": {
                "ward_id":         f"W{idx+1:02d}",
                "ward_name":       ward_names[idx],
                "flood_risk_base": risk_profile[idx],
                "population":      int(np.random.uniform(25000, 85000)),
                "area_sqkm":       round(np.random.uniform(3.2, 8.7), 2),
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon,     lat    ],
                    [lon + s, lat    ],
                    [lon + s, lat + s],
                    [lon,     lat + s],
                    [lon,     lat    ],
                ]],
            },
        })
    return wards


def generate_drain_network(wards: List[Dict]) -> List[Dict[str, Any]]:
    """Generate drain segments inside each ward polygon."""
    drains   = []
    drain_id = 1
    for ward in wards:
        coords     = ward["geometry"]["coordinates"][0]
        lat0_w     = coords[0][1]
        lon0_w     = coords[0][0]
        s          = WARD_SPACING
        flood_risk = ward["properties"]["flood_risk_base"]
        for _ in range(np.random.randint(3, 6)):
            slat = lat0_w + np.random.uniform(0.002, s - 0.002)
            slon = lon0_w + np.random.uniform(0.002, s - 0.002)
            elat = slat   + np.random.uniform(-0.008, 0.008)
            elon = slon   + np.random.uniform(0.002, 0.012)
            base_h = (1 - flood_risk) * 100
            health = float(np.clip(base_h + np.random.normal(0, 12), 5, 100))
            drains.append({
                "type": "Feature",
                "properties": {
                    "drain_id":          f"D{drain_id:03d}",
                    "ward_id":           ward["properties"]["ward_id"],
                    "drain_type":        np.random.choice(["open", "covered", "stormwater"]),
                    "age_years":         int(np.random.uniform(5, 35)),
                    "last_cleaned_days": int(np.random.uniform(10, 400)),
                    "health_score_base": round(health, 1),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[slon, slat], [elon, elat]],
                },
            })
            drain_id += 1
    return drains


def generate_road_network(
    city_lat: float = None,
    city_lon: float = None,
) -> List[Dict[str, Any]]:
    lat_c   = city_lat if city_lat is not None else _city_center[0]
    lon_c   = city_lon if city_lon is not None else _city_center[1]
    lat0    = lat_c - (GRID_ROWS / 2) * WARD_SPACING
    lon0    = lon_c - (GRID_COLS / 2) * WARD_SPACING
    roads   = []
    road_id = 1
    for r in range(GRID_ROWS + 1):
        lat = lat0 + r * WARD_SPACING
        for c in range(GRID_COLS):
            ls = lon0 + c * WARD_SPACING
            le = ls + WARD_SPACING
            roads.append({"road_id": f"R{road_id:03d}", "road_type": "arterial" if r % 2 == 0 else "secondary", "start": [ls, lat], "end": [le, lat]})
            road_id += 1
    for c in range(GRID_COLS + 1):
        lon = lon0 + c * WARD_SPACING
        for r in range(GRID_ROWS):
            ls = lat0 + r * WARD_SPACING
            le = ls + WARD_SPACING
            roads.append({"road_id": f"R{road_id:03d}", "road_type": "arterial" if c % 2 == 0 else "secondary", "start": [lon, ls], "end": [lon, le]})
            road_id += 1
    return roads


def generate_mock_dem() -> np.ndarray:
    np.random.seed(42)
    dem = np.random.normal(315, 12, (20, 20)).astype(np.float32)
    dem[6:10, 2:6]   -= 18
    dem[12:16, 8:13] -= 14
    return dem


def generate_rainfall_grid() -> np.ndarray:
    np.random.seed(7)
    base = np.random.normal(85, 22, (20, 20)).astype(np.float32)
    base[6:10, 2:6]   += 45
    base[12:16, 8:13] += 35
    return base
