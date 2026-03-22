"""
MODULE 6 — Digital Twin What-If Simulator (SIGNATURE FEATURE)
Interactive intervention sandbox — recalculates readiness + hotspots in real time.
"""

import numpy as np
from typing import Dict, Any, List
from .ward_readiness import WardReadiness
from .hotspot_engine import HotspotEngine


class DigitalTwin:
    """
    Recalculates readiness score and flood risk for a ward given
    intervention parameters. Designed for <3s latency.
    """

    def __init__(self):
        from .data_generator import _city_center
        lat, lon = _city_center
        self.ward_readiness = WardReadiness(city_lat=lat, city_lon=lon)
        self.hotspot_engine = HotspotEngine()
        self.hotspot_engine.set_city('default', lat, lon)
        # Pre-warm
        self._base_readiness = self.ward_readiness.get_ward_readiness_map()
        self._base_hotspots  = self.hotspot_engine.get_hotspots()

    def reinit(self, city_lat: float, city_lon: float) -> None:
        """Reinitialise for a new city."""
        self.ward_readiness   = WardReadiness(city_lat=city_lat, city_lon=city_lon)
        self.hotspot_engine   = HotspotEngine()
        self.hotspot_engine.set_city('default', city_lat, city_lon)
        self._base_readiness  = self.ward_readiness.get_ward_readiness_map()
        self._base_hotspots   = self.hotspot_engine.get_hotspots()

    # ── Intervention impact models ──────────────────────────────────────────

    def _drain_cleaning_effect(self, level: float) -> Dict[str, float]:
        """drain_cleaning ∈ [0,1] → readiness gain + flood risk reduction."""
        return {
            "readiness_gain": level * 14.0,
            "flood_risk_reduction": level * 0.22,
            "drain_health_boost": level * 25.0,
        }

    def _pump_deployment_effect(self, level: float) -> Dict[str, float]:
        return {
            "readiness_gain": level * 10.0,
            "flood_risk_reduction": level * 0.18,
            "drain_health_boost": level * 8.0,
        }

    def _new_drain_effect(self, level: float) -> Dict[str, float]:
        return {
            "readiness_gain": level * 20.0,
            "flood_risk_reduction": level * 0.30,
            "drain_health_boost": level * 35.0,
        }

    def _road_fix_effect(self, level: float) -> Dict[str, float]:
        return {
            "readiness_gain": level * 7.0,
            "flood_risk_reduction": level * 0.12,
            "drain_health_boost": level * 5.0,
        }

    def _compute_updated_hotspots(
        self,
        ward_id: str,
        flood_risk_reduction: float,
        ward_geometry
    ) -> List[Dict]:
        """
        Filter / downgrade hotspots that fall within the target ward geometry.
        Uses simple bounding box check for performance.
        """
        if ward_geometry is None:
            return self._base_hotspots["features"]

        # Get ward bounding box
        coords = ward_geometry["coordinates"][0]
        min_lon = min(c[0] for c in coords)
        max_lon = max(c[0] for c in coords)
        min_lat = min(c[1] for c in coords)
        max_lat = max(c[1] for c in coords)

        updated = []
        for feat in self._base_hotspots["features"]:
            geom = feat["geometry"]
            hcoords = geom["coordinates"][0]
            h_lat = sum(c[1] for c in hcoords) / len(hcoords)
            h_lon = sum(c[0] for c in hcoords) / len(hcoords)

            if min_lat <= h_lat <= max_lat and min_lon <= h_lon <= max_lon:
                # Downgrade risk score within this ward
                old_score = feat["properties"]["risk_score"]
                new_score = max(0.05, old_score * (1 - flood_risk_reduction))
                new_feat = {
                    **feat,
                    "properties": {
                        **feat["properties"],
                        "risk_score": round(new_score, 3),
                        "risk_category": self._reclassify(new_score),
                        "simulated": True,
                    }
                }
                updated.append(new_feat)
            else:
                updated.append(feat)

        return updated

    @staticmethod
    def _reclassify(score: float) -> str:
        if score >= 0.65:
            return "High"
        elif score >= 0.35:
            return "Medium"
        return "Low"

    # ── Main simulate endpoint ───────────────────────────────────────────────

    def simulate(
        self,
        ward_id: str,
        drain_cleaning: float,
        pump_deployment: float,
        new_drain: float,
        road_fix: float,
    ) -> Dict[str, Any]:

        # Validate
        for v in [drain_cleaning, pump_deployment, new_drain, road_fix]:
            if not 0 <= v <= 1:
                raise ValueError("Intervention values must be in [0, 1]")

        # Get base ward info
        if ward_id not in self._base_readiness:
            # Default to first ward
            ward_id = list(self._base_readiness.keys())[0]
        base = self._base_readiness[ward_id]

        # Aggregate effects
        effects = [
            self._drain_cleaning_effect(drain_cleaning),
            self._pump_deployment_effect(pump_deployment),
            self._new_drain_effect(new_drain),
            self._road_fix_effect(road_fix),
        ]

        total_readiness_gain = sum(e["readiness_gain"] for e in effects)
        total_flood_reduction = min(0.85, sum(e["flood_risk_reduction"] for e in effects))
        total_drain_boost = sum(e["drain_health_boost"] for e in effects)

        new_readiness = float(np.clip(base["readiness_score"] + total_readiness_gain, 0, 100))
        flood_risk_reduction_pct = round(total_flood_reduction * 100, 1)

        # Get ward geometry for hotspot update
        ward_geom = None
        for w in self.ward_readiness.wards:
            if w["properties"]["ward_id"] == ward_id:
                ward_geom = w["geometry"]
                break

        updated_hotspots = self._compute_updated_hotspots(ward_id, total_flood_reduction, ward_geom)

        # Count hotspot changes
        old_high = sum(1 for f in self._base_hotspots["features"]
                       if f["properties"]["risk_category"] == "High")
        new_high = sum(1 for f in updated_hotspots
                       if f["properties"]["risk_category"] == "High")

        return {
            "ward_id": ward_id,
            "ward_name": base["ward_name"],
            "baseline": {
                "readiness_score": base["readiness_score"],
                "risk_category": base["risk_category"],
                "drain_health": base["component_scores"]["drainage_health"],
            },
            "simulated": {
                "readiness_score": round(new_readiness, 1),
                "risk_category": self._reclassify_readiness(new_readiness),
                "drain_health": round(
                    min(100, base["component_scores"]["drainage_health"] + total_drain_boost), 1
                ),
            },
            "impact": {
                "readiness_gain": round(total_readiness_gain, 1),
                "flood_risk_reduction_pct": flood_risk_reduction_pct,
                "hotspots_downgraded": old_high - new_high,
                "drain_health_improvement": round(total_drain_boost, 1),
            },
            "updated_hotspots": {
                "type": "FeatureCollection",
                "features": updated_hotspots,
                "metadata": {
                    "high_risk_count": new_high,
                    "simulated": True,
                }
            },
            "interventions_applied": {
                "drain_cleaning": drain_cleaning,
                "pump_deployment": pump_deployment,
                "new_drain": new_drain,
                "road_fix": road_fix,
            }
        }

    @staticmethod
    def _reclassify_readiness(score: float) -> str:
        if score >= 80:
            return "Ready"
        elif score >= 60:
            return "Moderate"
        return "High Risk"
