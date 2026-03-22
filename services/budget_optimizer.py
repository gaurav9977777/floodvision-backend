"""
services/budget_optimizer.py  v2.0
=====================================
Realistic ward-level flood budget allocator powered by ConvLSTM risk scores.

What's new vs v1:
  ✦ Live flood risk from ConvLSTM / OpenWeather (replaces static flood_risk_base)
  ✦ Explicit per-ward budget allocations shown in output
  ✦ Three allocation modes: optimal | equity | triage
  ✦ Sensitivity analysis (marginal value of ₹10L more)
  ✦ Equity metrics (Gini coefficient)
  ✦ Implementation timeline (Immediate / Pre-monsoon / Medium-term)
  ✦ AMRUT 2.0 component tagging
"""

import math, datetime
from typing import Dict, Any, List, Tuple, Optional
from .ward_readiness    import WardReadiness
from .drain_intelligence import DrainIntelligence
from .flood_risk_provider import get_ward_risk_scores

# ── Catalogue ────────────────────────────────────────────────────────────────
INTERVENTIONS: List[Dict] = [
    {"id":"drain_desilting","name":"Primary Drain Desilting & Cleaning",
     "desc":"Full mechanical desilting of primary stormwater drains, removal of encroachments",
     "cost_inr":750_000,"readiness_gain":14.0,"risk_reduction_pct":20.0,"lives_per_lakh":2.1,
     "duration_days":7,"category":"Drainage","urgency":1.4,"min_dtm":14,"bcr":4.2,
     "sdg":"SDG 11.5","amrut":"Stormwater Drainage","risk_floor":0.30},
    {"id":"pump_deployment","name":"Submersible Pump Deployment (6 units)",
     "desc":"6×100HP pumps with operators at critical low points",
     "cost_inr":1_400_000,"readiness_gain":11.0,"risk_reduction_pct":17.0,"lives_per_lakh":3.8,
     "duration_days":3,"category":"Emergency Equipment","urgency":1.5,"min_dtm":7,"bcr":3.6,
     "sdg":"SDG 13.1","amrut":"Emergency Response","risk_floor":0.50},
    {"id":"nala_dredging","name":"Nala Dredging & Bank Reinforcement",
     "desc":"Mechanical dredging of natural nallahs, HDPE lining and bank protection",
     "cost_inr":2_200_000,"readiness_gain":20.0,"risk_reduction_pct":30.0,"lives_per_lakh":5.2,
     "duration_days":21,"category":"Drainage","urgency":1.2,"min_dtm":30,"bcr":5.8,
     "sdg":"SDG 11.5","amrut":"Stormwater Drainage","risk_floor":0.55},
    {"id":"stormwater_aug","name":"Stormwater Drain Augmentation (500m RCC box drain)",
     "desc":"New RCC box drain construction to increase peak flow capacity",
     "cost_inr":4_200_000,"readiness_gain":25.0,"risk_reduction_pct":35.0,"lives_per_lakh":6.0,
     "duration_days":45,"category":"Infrastructure","urgency":0.8,"min_dtm":60,"bcr":6.1,
     "sdg":"SDG 11.1","amrut":"Stormwater Drainage","risk_floor":0.65},
    {"id":"culvert_repair","name":"Culvert Repair & Capacity Upgrade",
     "desc":"Structural repair + hydraulic capacity upgrade of damaged culverts",
     "cost_inr":680_000,"readiness_gain":8.5,"risk_reduction_pct":13.0,"lives_per_lakh":1.8,
     "duration_days":14,"category":"Infrastructure","urgency":1.1,"min_dtm":21,"bcr":3.9,
     "sdg":"SDG 11.1","amrut":"Stormwater Drainage","risk_floor":0.35},
    {"id":"iot_sensor","name":"IoT Flood Sensor Network (15 nodes)",
     "desc":"Water-level + rain-gauge sensors with real-time ICCC dashboard integration",
     "cost_inr":540_000,"readiness_gain":6.0,"risk_reduction_pct":8.0,"lives_per_lakh":4.5,
     "duration_days":5,"category":"Technology","urgency":1.3,"min_dtm":10,"bcr":7.2,
     "sdg":"SDG 13.3","amrut":"Smart City / ICCC","risk_floor":0.0},
    {"id":"early_warning","name":"Community Early Warning System",
     "desc":"SMS alert integration, siren network, ward-level evacuation plan activation",
     "cost_inr":280_000,"readiness_gain":7.0,"risk_reduction_pct":10.0,"lives_per_lakh":8.0,
     "duration_days":3,"category":"Technology","urgency":1.5,"min_dtm":5,"bcr":9.1,
     "sdg":"SDG 13.1","amrut":"Smart City / ICCC","risk_floor":0.0},
    {"id":"road_bunding","name":"Critical Road Elevation Bunds",
     "desc":"Temporary/permanent bunds on key evacuation routes prone to waterlogging",
     "cost_inr":420_000,"readiness_gain":5.5,"risk_reduction_pct":9.0,"lives_per_lakh":2.0,
     "duration_days":10,"category":"Roads","urgency":1.2,"min_dtm":15,"bcr":3.2,
     "sdg":"SDG 9.1","amrut":"Urban Mobility","risk_floor":0.40},
    {"id":"manhole_safety","name":"Manhole Audit, Locking & Replacement",
     "desc":"Full ward manhole inventory, broken cover replacement, anti-displacement locking",
     "cost_inr":160_000,"readiness_gain":3.5,"risk_reduction_pct":5.0,"lives_per_lakh":6.0,
     "duration_days":4,"category":"Maintenance","urgency":1.4,"min_dtm":7,"bcr":5.5,
     "sdg":"SDG 3.6","amrut":"Maintenance","risk_floor":0.0},
    {"id":"retention_pond","name":"Detention / Retention Pond Construction",
     "desc":"New or restored water detention basin to absorb peak runoff",
     "cost_inr":8_500_000,"readiness_gain":30.0,"risk_reduction_pct":40.0,"lives_per_lakh":7.5,
     "duration_days":90,"category":"Infrastructure","urgency":0.6,"min_dtm":120,"bcr":4.8,
     "sdg":"SDG 11.5","amrut":"Stormwater Drainage","risk_floor":0.70},
    {"id":"relief_prepos","name":"Emergency Relief Material Pre-positioning",
     "desc":"NDRF-standard relief kits, medicines, boats at ward-level stores",
     "cost_inr":350_000,"readiness_gain":4.5,"risk_reduction_pct":6.0,"lives_per_lakh":5.0,
     "duration_days":2,"category":"Emergency Equipment","urgency":1.6,"min_dtm":3,"bcr":4.9,
     "sdg":"SDG 13.1","amrut":"Emergency Response","risk_floor":0.0},
    {"id":"green_infra","name":"Green Infrastructure (Permeable Paving + Bioswales)",
     "desc":"Permeable pavements, bioswales, rain gardens — reduce surface runoff 25-30%",
     "cost_inr":1_800_000,"readiness_gain":10.0,"risk_reduction_pct":15.0,"lives_per_lakh":2.8,
     "duration_days":30,"category":"Green Infrastructure","urgency":0.7,"min_dtm":45,"bcr":3.4,
     "sdg":"SDG 15.1","amrut":"Green & Blue Infrastructure","risk_floor":0.30},
]

CATEGORY_CAP   = 0.45
REPEAT_PENALTY = 0.55
MIN_GAIN       = 1.0
EQUITY_FLOOR   = 0.03

MODES = {
    "optimal": "ROI-maximising greedy knapsack",
    "equity":  "Floor allocation to every ward then ROI top-up",
    "triage":  "80% to top-5 highest-risk wards, 20% spread",
}


class BudgetOptimizer:

    def __init__(self, city_lat=None, city_lon=None):
        from .data_generator import _city_center
        lat = city_lat or _city_center[0]
        lon = city_lon or _city_center[1]
        self._city_lat = lat
        self._city_lon = lon
        self._wr = WardReadiness(city_lat=lat, city_lon=lon)
        self._di = DrainIntelligence(city_lat=lat, city_lon=lon)

    def reinit(self, city_lat: float, city_lon: float) -> None:
        """Reinitialise for a new city."""
        self._city_lat = city_lat
        self._city_lon = city_lon
        self._wr = WardReadiness(city_lat=city_lat, city_lon=city_lon)
        self._di = DrainIntelligence(city_lat=city_lat, city_lon=city_lon)

    @staticmethod
    def _dtm():
        today   = datetime.date.today()
        monsoon = datetime.date(today.year, 6, 1)
        if today > monsoon:
            monsoon = datetime.date(today.year + 1, 6, 1)
        return (monsoon - today).days

    @staticmethod
    def _pop_w(pop):
        return math.log10(max(pop, 1000)) / math.log10(100_000)

    @staticmethod
    def _gini(vals):
        if not vals or sum(vals) == 0: return 0.0
        a = sorted(vals); n = len(a)
        return round((2*sum((i+1)*v for i,v in enumerate(a)) - (n+1)*sum(a)) / (n*sum(a)), 3)

    @staticmethod
    def _fmt(x):
        if x >= 1e7: return f"₹{x/1e7:.2f}Cr"
        if x >= 1e5: return f"₹{x/1e5:.1f}L"
        return f"₹{x:,.0f}"

    def _envelopes(self, wards, budget, mode):
        if mode == "equity":
            floor = budget * EQUITY_FLOOR
            rem   = max(0, budget - floor * len(wards))
            ws    = {w["ward_id"]: w["live_risk"] * w["pop_w"] for w in wards}
            tw    = sum(ws.values()) or 1
            return {w["ward_id"]: floor + rem * ws[w["ward_id"]] / tw for w in wards}
        elif mode == "triage":
            top5  = sorted(wards, key=lambda x: x["live_risk"], reverse=True)[:5]
            top_i = {w["ward_id"] for w in top5}
            rest  = [w for w in wards if w["ward_id"] not in top_i]
            tw    = sum(w["live_risk"] for w in top5) or 1
            env   = {w["ward_id"]: budget * 0.80 * w["live_risk"] / tw for w in top5}
            for w in rest:
                env[w["ward_id"]] = (budget * 0.20 / len(rest)) if rest else 0
            return env
        else:
            return {w["ward_id"]: budget for w in wards}

    def optimize(self, total_budget: float, mode: str = "optimal", **_) -> Dict[str, Any]:
        dtm      = self._dtm()
        ward_map = self._wr.get_ward_readiness_map()
        drain_map = self._di.get_ward_avg_health()

        from .data_generator import generate_ward_boundaries
        geoms = {f["properties"]["ward_id"]: f for f in generate_ward_boundaries(city_lat=self._city_lat, city_lon=self._city_lon)}

        # Build lat/lon centroids
        ext = {}
        for wid, info in ward_map.items():
            feat = geoms.get(wid, {})
            coords = feat.get("geometry", {}).get("coordinates", [[]])[0]
            if coords:
                clat = sum(c[1] for c in coords) / len(coords)
                clon = sum(c[0] for c in coords) / len(coords)
            else:
                clat, clon = self._city_lat, self._city_lon
            ext[wid] = {**info, "lat": clat, "lon": clon,
                        "flood_risk_base": feat.get("properties", {}).get("flood_risk_base", 0.5)}

        live = get_ward_risk_scores(ext, drain_map, city_avg_elev=315.0)

        # Enrich ward list
        wards = []
        for wid, info in ward_map.items():
            pop   = info.get("population", 15000)
            rdy   = info.get("readiness_score", 50.0)
            lr    = live.get(wid, {})
            risk  = lr.get("risk_score", info.get("flood_risk_base", 0.5))
            area  = geoms.get(wid, {}).get("properties", {}).get("area_sqkm", 5.0)
            vuln  = risk * math.log10(max(pop, 1000)) * math.log(max(area, 1.0) + 1)
            wards.append({
                "ward_id": wid, "ward_name": info["ward_name"],
                "readiness": rdy, "population": pop, "area_sqkm": area,
                "live_risk": risk, "risk_source": lr.get("source", "static"),
                "vulnerability": round(vuln, 3), "pop_w": self._pop_w(pop),
                "drain_health": drain_map.get(wid, 60.0),
                "daily_rain": lr.get("daily_rainfall_mm", []),
                "components": {
                    "model": lr.get("model_score"), "rainfall": lr.get("rainfall_score"),
                    "elevation": lr.get("elevation_score"), "drainage": lr.get("drain_score"),
                },
            })

        wards.sort(key=lambda w: w["vulnerability"], reverse=True)
        envs = self._envelopes(wards, total_budget, mode)

        # Greedy allocation
        candidates = []
        for ward in wards[:20]:
            for intv in INTERVENTIONS:
                if intv["min_dtm"] > dtm: continue
                if ward["live_risk"] < intv["risk_floor"]: continue
                rb  = 1.0 + ward["live_risk"] * 0.6
                eff = intv["readiness_gain"] * intv["urgency"] * ward["pop_w"] * intv["bcr"] * rb / intv["cost_inr"] * 1e6
                candidates.append((eff, {"ward": ward, "intv": intv, "eff": round(eff, 4)}))
        candidates.sort(reverse=True)

        allocated, remaining = [], total_budget
        cat_spend, ward_spend, ward_cnt = {}, {}, {}
        total_rg = total_rr = total_lives = 0.0

        for _, item in candidates:
            ward, intv = item["ward"], item["intv"]
            wid, cost  = ward["ward_id"], intv["cost_inr"]
            if remaining < cost: continue
            if cat_spend.get(intv["category"], 0) + cost > total_budget * CATEGORY_CAP: continue
            if ward_spend.get(wid, 0) + cost > envs.get(wid, total_budget): continue
            wic     = ward_cnt.setdefault(wid, {})
            rep     = wic.get(intv["id"], 0)
            gm      = REPEAT_PENALTY ** rep
            eg      = intv["readiness_gain"] * gm
            if eg < MIN_GAIN: continue

            remaining -= cost
            cat_spend[intv["category"]] = cat_spend.get(intv["category"], 0) + cost
            ward_spend[wid]             = ward_spend.get(wid, 0) + cost
            wic[intv["id"]]            = rep + 1
            total_rg    += eg
            total_rr    += intv["risk_reduction_pct"] * gm
            total_lives += intv["lives_per_lakh"] * ward["population"] / 100_000 * gm

            allocated.append({
                "ward_id": wid, "ward_name": ward["ward_name"],
                "current_readiness": round(ward["readiness"], 1),
                "population": ward["population"], "area_sqkm": ward["area_sqkm"],
                "vulnerability_score": ward["vulnerability"],
                "live_risk_score": ward["live_risk"], "risk_source": ward["risk_source"],
                "daily_rainfall_mm": ward["daily_rain"],
                "drain_health": round(ward["drain_health"], 1),
                "component_risk_scores": ward["components"],
                "intervention": intv["name"], "intervention_id": intv["id"],
                "description": intv["desc"], "category": intv["category"],
                "sdg_goal": intv["sdg"], "amrut_component": intv["amrut"],
                "cost_inr": cost, "cost_display": self._fmt(cost),
                "readiness_gain": round(eg, 1),
                "projected_readiness": round(min(100, ward["readiness"] + eg), 1),
                "flood_risk_reduction_pct": round(intv["risk_reduction_pct"] * gm, 1),
                "lives_protected_est": round(intv["lives_per_lakh"] * ward["population"] / 100_000 * gm, 1),
                "duration_days": intv["duration_days"], "benefit_cost_ratio": intv["bcr"],
                "urgency_factor": intv["urgency"], "efficiency_score": round(item["eff"], 3),
                "priority_rank": len(allocated) + 1, "days_to_monsoon": dtm,
            })

        # Per-ward allocation summary
        wa: Dict[str, dict] = {}
        for a in allocated:
            wid = a["ward_id"]
            if wid not in wa:
                wa[wid] = {
                    "ward_id": wid, "ward_name": a["ward_name"],
                    "population": a["population"], "area_sqkm": a["area_sqkm"],
                    "live_risk_score": a["live_risk_score"], "risk_source": a["risk_source"],
                    "daily_rainfall_mm": a["daily_rainfall_mm"],
                    "drain_health": a["drain_health"],
                    "current_readiness": a["current_readiness"],
                    "vulnerability": a["vulnerability_score"],
                    "component_risk_scores": a["component_risk_scores"],
                    "total_allocated_inr": 0, "total_readiness_gain": 0.0,
                    "total_risk_reduction": 0.0, "total_lives_protected": 0.0,
                    "interventions": [], "categories": [],
                }
            wa[wid]["total_allocated_inr"]   += a["cost_inr"]
            wa[wid]["total_readiness_gain"]  += a["readiness_gain"]
            wa[wid]["total_risk_reduction"]  += a["flood_risk_reduction_pct"]
            wa[wid]["total_lives_protected"] += a["lives_protected_est"]
            wa[wid]["interventions"].append(a["intervention"])
            if a["category"] not in wa[wid]["categories"]:
                wa[wid]["categories"].append(a["category"])

        ward_list = []
        for ws in wa.values():
            ws["total_allocated_display"]  = self._fmt(ws["total_allocated_inr"])
            ws["total_readiness_gain"]     = round(ws["total_readiness_gain"], 1)
            ws["projected_readiness"]      = round(min(100, ws["current_readiness"] + ws["total_readiness_gain"]), 1)
            ws["total_risk_reduction"]     = round(min(90, ws["total_risk_reduction"]), 1)
            ws["total_lives_protected"]    = round(ws["total_lives_protected"], 1)
            ws["intervention_count"]       = len(ws["interventions"])
            ws["allocation_pct_of_budget"] = round(ws["total_allocated_inr"] / total_budget * 100, 1)
            ws["budget_per_capita_inr"]    = round(ws["total_allocated_inr"] / max(ws["population"], 1), 0)
            ws["roi_score"]                = round(ws["total_readiness_gain"] * ws["population"] / max(ws["total_allocated_inr"], 1) * 1e5, 2)
            ward_list.append(ws)
        ward_list.sort(key=lambda x: x["total_allocated_inr"], reverse=True)

        # Category breakdown
        cat_totals: dict = {}
        for a in allocated:
            cat = a["category"]
            if cat not in cat_totals:
                cat_totals[cat] = {"budget_inr": 0, "risk_reduction_pct": 0, "count": 0, "amrut": a["amrut_component"]}
            cat_totals[cat]["budget_inr"]         += a["cost_inr"]
            cat_totals[cat]["risk_reduction_pct"] += a["flood_risk_reduction_pct"]
            cat_totals[cat]["count"]              += 1

        # Sensitivity
        extra = 10_00_000
        extra_rr2 = 0.0
        for ward in wards[:10]:
            for intv in sorted(INTERVENTIONS, key=lambda x: x["bcr"], reverse=True):
                if intv["min_dtm"] > dtm or intv["cost_inr"] > extra: continue
                wic = ward_cnt.get(ward["ward_id"], {})
                rp  = wic.get(intv["id"], 0)
                extra_rr2 += intv["risk_reduction_pct"] * (REPEAT_PENALTY ** rp)
                extra -= intv["cost_inr"]
                if extra <= 0: break
            if extra <= 0: break

        # Timeline
        today   = datetime.date.today()
        seen    = set()
        phases  = {"Immediate (≤7 days)": [], "Pre-monsoon (8–30 days)": [], "Medium-term (>30 days)": []}
        for a in allocated:
            k = a["intervention"]
            if k in seen: continue
            seen.add(k)
            end = today + datetime.timedelta(days=a["duration_days"])
            entry = {
                "intervention": k, "category": a["category"],
                "duration_days": a["duration_days"],
                "start_date": today.strftime("%b %d"), "end_date": end.strftime("%b %d"),
                "urgency": a["urgency_factor"], "total_cost_inr": a["cost_inr"],
            }
            if a["urgency_factor"] >= 1.4:
                phases["Immediate (≤7 days)"].append(entry)
            elif a["duration_days"] <= 30:
                phases["Pre-monsoon (8–30 days)"].append(entry)
            else:
                phases["Medium-term (>30 days)"].append(entry)

        budget_used = total_budget - remaining

        return {
            "total_budget_inr": total_budget, "budget_utilized_inr": budget_used,
            "budget_remaining_inr": remaining,
            "utilization_pct": round(budget_used / total_budget * 100, 1) if total_budget else 0,
            "days_to_monsoon": dtm, "mode": mode, "mode_description": MODES.get(mode, mode),
            "interventions": allocated,
            "ward_allocations": ward_list,
            "risk_reduction_by_category": [
                {"category": cat, "risk_reduction_pct": round(v["risk_reduction_pct"], 1),
                 "budget_inr": v["budget_inr"], "budget_display": self._fmt(v["budget_inr"]),
                 "intervention_count": v["count"], "amrut_component": v["amrut"]}
                for cat, v in sorted(cat_totals.items(), key=lambda x: x[1]["risk_reduction_pct"], reverse=True)
            ],
            "sensitivity_analysis": {
                "additional_budget_inr": 10_00_000, "additional_budget_display": "₹10L",
                "marginal_risk_reduction_pct": round(extra_rr2, 2),
                "interpretation": f"Each additional ₹10L reduces flood risk by ~{extra_rr2:.1f}% more across priority wards",
            },
            "equity_metrics": {
                "gini_coefficient": self._gini([ws["total_allocated_inr"] for ws in ward_list]),
                "gini_interpretation": (
                    "High equity" if self._gini([ws["total_allocated_inr"] for ws in ward_list]) < 0.3
                    else "Moderate equity" if self._gini([ws["total_allocated_inr"] for ws in ward_list]) < 0.55
                    else "Concentrated (risk-targeted)"
                ),
                "ward_count_allocated": len(ward_list),
                "ward_count_total": len(wards),
                "coverage_pct": round(len(ward_list) / max(len(wards), 1) * 100, 1),
                "budget_per_capita_avg": round(budget_used / max(sum(w["population"] for w in wards), 1), 0),
            },
            "implementation_timeline": [
                {"phase": ph, "items": items, "item_count": len(items)}
                for ph, items in phases.items()
            ],
            "summary": {
                "total_readiness_gain": round(total_rg, 1),
                "total_risk_reduction_pct": round(min(total_rr, 90), 1),
                "total_lives_protected_est": round(total_lives, 1),
                "wards_covered": len(ward_list), "interventions_count": len(allocated),
                "avg_benefit_cost_ratio": round(sum(a["benefit_cost_ratio"] for a in allocated) / len(allocated), 2) if allocated else 0,
            },
            "total_readiness_gain": round(total_rg, 1),
            "total_risk_reduction_pct": round(min(total_rr, 90), 1),
            "wards_covered": len(ward_list),
            "data_sources": [
                "ConvLSTM Rainfall Model (NASA GPM IMERG)",
                "OpenWeather API (real-time rainfall)",
                "NDMA Urban Flood Guidelines 2022",
                "AMRUT 2.0 Stormwater Guidelines 2023",
                "MoUD Stormwater Drainage Manual 2019",
                "Smart Cities Mission PAC 2023-24",
            ],
        }
