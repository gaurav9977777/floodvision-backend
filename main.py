"""
FloodVision-Pro — FastAPI Backend
Key fix: every endpoint that depends on geographic data calls set_city_center()
FIRST so all generated ward/drain/hotspot coordinates are anchored to the
actual requested city, not the hardcoded Nagpur default.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from services.data_generator import set_city_center
from services.hotspot_engine  import HotspotEngine
from services.drain_intelligence import DrainIntelligence
from services.ward_readiness  import WardReadiness
from services.route_engine    import RouteEngine
from services.budget_optimizer import BudgetOptimizer
from services.digital_twin    import DigitalTwin
from services.chatbot         import chat as groq_chat, get_suggested_questions

app = FastAPI(
    title="FloodVision-Pro API",
    description="Dynamic city-aware flood intelligence platform",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singleton services (re-initialised per city via set_city) ─────────────
hotspot_engine   = HotspotEngine()
drain_intel      = DrainIntelligence()
ward_ready       = WardReadiness()
route_engine     = RouteEngine()
budget_optimizer = BudgetOptimizer()
digital_twin     = DigitalTwin()

# Tracks the last city so we only reinit when the city changes
_last_city: Dict = {}


def _switch_city(city_id: str, lat: float, lon: float) -> None:
    """
    Reinitialise all services when the city changes.
    This ensures wards, drains, hotspots are all at the correct location.
    """
    global _last_city
    if _last_city.get("id") == city_id and \
       abs(_last_city.get("lat", 0) - lat) < 0.001 and \
       abs(_last_city.get("lon", 0) - lon) < 0.001:
        return  # same city — no reinit needed

    # Update city centre for data generator (all coordinates follow)
    set_city_center(lat, lon)

    # Reinit each service so it regenerates data for the new city
    hotspot_engine.set_city(city_id, lat, lon)
    drain_intel.reinit(lat, lon)
    ward_ready.reinit(lat, lon)
    route_engine.reinit(lat, lon)
    digital_twin.reinit(lat, lon)
    budget_optimizer.reinit(lat, lon)

    _last_city = {"id": city_id, "lat": lat, "lon": lon}


# ── Request models ────────────────────────────────────────────────────────

class RouteRequest(BaseModel):
    source_lat: float
    source_lon: float
    dest_lat:   float
    dest_lon:   float

class BudgetRequest(BaseModel):
    total_budget: float
    mode: str = "optimal"

class SimulateRequest(BaseModel):
    ward_id:        str
    drain_cleaning: float = 0.0
    pump_deployment:float = 0.0
    new_drain:      float = 0.0
    road_fix:       float = 0.0

class ChatMessage(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    history:   List[ChatMessage] = []
    message:   str
    city_data: Optional[Dict[str, Any]] = None
    fast_mode: bool = False


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "operational", "system": "FloodVision-Pro v2.0"}


@app.get("/hotspots")
def get_hotspots(
    city_id: str   = Query(default="bhilai"),
    lat:     float = Query(default=21.209),
    lon:     float = Query(default=81.378),
):
    """Flood hotspot polygons centred on the requested city."""
    _switch_city(city_id, lat, lon)
    return hotspot_engine.get_hotspots()


@app.get("/ward-readiness")
def get_ward_readiness(
    city_id: str   = Query(default="bhilai"),
    lat:     float = Query(default=21.209),
    lon:     float = Query(default=81.378),
):
    """Ward readiness scores for the requested city."""
    _switch_city(city_id, lat, lon)
    return ward_ready.get_readiness()


@app.get("/drain-health")
def get_drain_health(
    city_id: str   = Query(default="bhilai"),
    lat:     float = Query(default=21.209),
    lon:     float = Query(default=81.378),
):
    """Drain health index for the requested city."""
    _switch_city(city_id, lat, lon)
    return drain_intel.get_drain_health()


@app.post("/safe-route")
def compute_safe_route(req: RouteRequest):
    return route_engine.compute_safe_route(
        req.source_lat, req.source_lon,
        req.dest_lat,   req.dest_lon,
    )


@app.post("/optimize-budget")
def optimize_budget(req: BudgetRequest):
    return budget_optimizer.optimize(req.total_budget, mode=req.mode)


@app.get("/ward-allocations")
def ward_allocations(budget: float = 10_000_000, mode: str = "optimal"):
    result = budget_optimizer.optimize(budget, mode=mode)
    return {
        "ward_allocations":  result["ward_allocations"],
        "equity_metrics":    result["equity_metrics"],
        "sensitivity":       result["sensitivity_analysis"],
        "timeline":          result["implementation_timeline"],
        "total_budget_inr":  budget,
        "mode":              mode,
    }


@app.post("/simulate")
def simulate(req: SimulateRequest):
    return digital_twin.simulate(
        ward_id=        req.ward_id,
        drain_cleaning= req.drain_cleaning,
        pump_deployment=req.pump_deployment,
        new_drain=      req.new_drain,
        road_fix=       req.road_fix,
    )


@app.get("/health")
def health():
    return {"status": "healthy", "modules": 6, "city": _last_city}


@app.post("/chat")
def chatbot(req: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in req.history]
    reply   = groq_chat(
        history=history,
        user_message=req.message,
        city_data=req.city_data,
        fast_mode=req.fast_mode,
    )
    return {"reply": reply, "model": "llama-3.3-70b-versatile"}


@app.get("/chat/suggestions/{city_name}")
def chat_suggestions(city_name: str):
    return {"suggestions": get_suggested_questions(city_name)}
