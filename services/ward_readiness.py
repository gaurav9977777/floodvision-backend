"""
MODULE 3 — Ward Pre-Monsoon Readiness Score
Composite weighted metric for municipal decision-making.
"""

import numpy as np
from typing import Dict, Any, List
from .data_generator import generate_ward_boundaries
from .drain_intelligence import DrainIntelligence


PRIORITY_ACTIONS_BANK = {
    "drain_clean": "Emergency drain desiltation of primary channels",
    "pump_deploy": "Deploy submersible pumps at identified low-lying points",
    "culvert_repair": "Repair blocked culverts on arterial roads",
    "road_elevation": "Temporary road elevation bunds at flood entry points",
    "early_warning": "Install IoT water-level sensors in critical drains",
    "community_drill": "Conduct ward-level flood evacuation drill",
    "nala_dredging": "Nala dredging & bank reinforcement",
    "manhole_audit": "Manhole cover integrity audit & replacement",
    "stormwater_infra": "Fast-track stormwater infrastructure augmentation",
    "mapping": "Complete micro-topography survey for drainage master plan",
}

SCORE_WEIGHTS = {
    "drainage_health": 0.30,
    "flood_exposure": 0.25,
    "historical_flooding": 0.20,
    "accessibility": 0.15,
    "infrastructure_vulnerability": 0.10,
}


class WardReadiness:
    """Computes pre-monsoon readiness score per ward."""

    def __init__(self, city_lat=None, city_lon=None):
        from .data_generator import _city_center
        lat = city_lat or _city_center[0]
        lon = city_lon or _city_center[1]
        self.wards            = generate_ward_boundaries(city_lat=lat, city_lon=lon)
        self.drain_intel      = DrainIntelligence(city_lat=lat, city_lon=lon)
        self._readiness_cache: Dict[str, Dict] = {}

    def reinit(self, city_lat: float, city_lon: float) -> None:
        """Regenerate ward data for a new city centre."""
        self.wards            = generate_ward_boundaries(city_lat=city_lat, city_lon=city_lon)
        self.drain_intel      = DrainIntelligence(city_lat=city_lat, city_lon=city_lon)
        self._readiness_cache = {}

    def _mock_historical_score(self, ward_id: str, flood_risk: float) -> float:
        """Mock historical flood frequency → score 0–100 (100 = never flooded)."""
        # Correlated with flood risk but with variance
        np.random.seed(hash(ward_id) % 1000)
        h = (1 - flood_risk) * 100 + np.random.normal(0, 8)
        return float(np.clip(h, 10, 100))

    def _accessibility_score(self, ward) -> float:
        """Proxy: area and population density → accessibility."""
        area = ward["properties"]["area_sqkm"]
        pop = ward["properties"]["population"]
        density = pop / area  # persons/km²
        # Higher density = more roads = better accessibility
        score = min(100, 40 + density / 800)
        return float(score)

    def _infra_vulnerability(self, flood_risk: float, avg_drain_health: float) -> float:
        """Infrastructure vulnerability = 100 - f(risk, drain_health)."""
        vuln = 100 - (flood_risk * 50 + (100 - avg_drain_health) * 0.5)
        return float(np.clip(vuln, 0, 100))

    @staticmethod
    def _risk_category(score: float) -> str:
        if score >= 80:
            return "Ready"
        elif score >= 60:
            return "Moderate"
        return "High Risk"

    def _top_actions(self, score: float, flood_risk: float, drain_health: float) -> List[str]:
        """Return top 3 priority actions based on gaps."""
        candidates = []
        if drain_health < 60:
            candidates += ["drain_clean", "manhole_audit"]
        if flood_risk > 0.7:
            candidates += ["pump_deploy", "road_elevation", "nala_dredging"]
        if score < 60:
            candidates += ["early_warning", "community_drill"]
        if flood_risk > 0.5:
            candidates += ["culvert_repair", "stormwater_infra"]
        candidates += ["mapping"]
        seen = []
        for c in candidates:
            if c not in seen:
                seen.append(c)
        return [PRIORITY_ACTIONS_BANK[k] for k in seen[:3]]

    def compute_readiness(self) -> List[Dict[str, Any]]:
        drain_health_map = self.drain_intel.get_ward_avg_health()
        results = []

        for ward in self.wards:
            wid = ward["properties"]["ward_id"]
            flood_risk = ward["properties"]["flood_risk_base"]

            drain_h = drain_health_map.get(wid, 60.0)
            hist_h = self._mock_historical_score(wid, flood_risk)
            access_h = self._accessibility_score(ward)
            infra_h = self._infra_vulnerability(flood_risk, drain_h)

            # Composite readiness score
            readiness = (
                SCORE_WEIGHTS["drainage_health"] * drain_h +
                SCORE_WEIGHTS["flood_exposure"] * (1 - flood_risk) * 100 +
                SCORE_WEIGHTS["historical_flooding"] * hist_h +
                SCORE_WEIGHTS["accessibility"] * access_h +
                SCORE_WEIGHTS["infrastructure_vulnerability"] * infra_h
            )
            # Add base offset so wards aren't uniformly low (realistic municipal range)
            readiness = readiness + 20.0
            readiness = float(np.clip(readiness, 0, 100))
            cat = self._risk_category(readiness)
            actions = self._top_actions(readiness, flood_risk, drain_h)

            feat = {
                "type": "Feature",
                "properties": {
                    "ward_id": wid,
                    "ward_name": ward["properties"]["ward_name"],
                    "readiness_score": round(readiness, 1),
                    "risk_category": cat,
                    "priority_actions": actions,
                    "component_scores": {
                        "drainage_health": round(drain_h, 1),
                        "flood_exposure": round((1 - flood_risk) * 100, 1),
                        "historical_flooding": round(hist_h, 1),
                        "accessibility": round(access_h, 1),
                        "infrastructure_vulnerability": round(infra_h, 1),
                    },
                    "population": ward["properties"]["population"],
                    "area_sqkm": ward["properties"]["area_sqkm"],
                    "flood_risk_base": ward["properties"].get("flood_risk_base", 0.5),
                },
                "geometry": ward["geometry"],
            }
            results.append(feat)
            self._readiness_cache[wid] = feat["properties"]

        return results

    def get_readiness(self) -> Dict[str, Any]:
        features = self.compute_readiness()
        cats = [f["properties"]["risk_category"] for f in features]
        avg_score = np.mean([f["properties"]["readiness_score"] for f in features])
        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "total_wards": len(features),
                "ready": cats.count("Ready"),
                "moderate": cats.count("Moderate"),
                "high_risk": cats.count("High Risk"),
                "city_avg_readiness": round(float(avg_score), 1),
            }
        }

    def get_ward_readiness_map(self) -> Dict[str, Dict]:
        """Utility: ward_id → readiness properties dict."""
        if not self._readiness_cache:
            self.compute_readiness()
        return self._readiness_cache
