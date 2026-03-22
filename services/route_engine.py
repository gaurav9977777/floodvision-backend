"""
MODULE 4 — Flood-Aware Critical Route Engine
Builds road graph using NetworkX, penalizes edges near flood hotspots,
and computes the safest (not shortest) path.
"""

import math
import networkx as nx
import numpy as np
from typing import Dict, Any, List, Tuple
from .data_generator import (
    generate_road_network, CITY_CENTER, WARD_SPACING, GRID_ROWS, GRID_COLS
)


def haversine(lat1, lon1, lat2, lon2) -> float:
    """Return distance in metres between two lat/lon points."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


class RouteEngine:
    """Flood-aware route planner on a synthetic road graph."""

    FLOOD_PENALTY_RADIUS_M = 300   # metres
    FLOOD_WEIGHT_MULTIPLIER = 8.0  # apply to edges within penalty radius
    SPEED_KMPH = 25.0              # assumed urban speed

    def __init__(self, city_lat=None, city_lon=None):
        from .data_generator import _city_center
        lat = city_lat or _city_center[0]
        lon = city_lon or _city_center[1]
        self.roads       = generate_road_network(city_lat=lat, city_lon=lon)
        self.graph       = self._build_graph()
        self._flood_cells: List[Tuple[float, float]] = []

    def reinit(self, city_lat: float, city_lon: float) -> None:
        """Rebuild road graph for a new city centre."""
        self.roads       = generate_road_network(city_lat=city_lat, city_lon=city_lon)
        self.graph       = self._build_graph()
        self._flood_cells = []

    def _build_graph(self) -> nx.Graph:
        G = nx.Graph()
        for road in self.roads:
            s = road["start"]   # [lon, lat]
            e = road["end"]     # [lon, lat]
            s_key = (round(s[1], 5), round(s[0], 5))
            e_key = (round(e[1], 5), round(e[0], 5))
            dist = haversine(s[1], s[0], e[1], e[0])
            G.add_edge(s_key, e_key,
                       weight=dist,
                       road_id=road["road_id"],
                       road_type=road["road_type"],
                       lon_s=s[0], lat_s=s[1],
                       lon_e=e[0], lat_e=e[1])
        return G

    def set_flood_zones(self, flood_cells: List[Tuple[float, float]]):
        """Inject high-risk lat/lon cells to penalize nearby edges."""
        self._flood_cells = flood_cells
        self._apply_flood_penalties()

    def _apply_flood_penalties(self):
        if not self._flood_cells:
            return
        for u, v, data in self.graph.edges(data=True):
            mid_lat = (data["lat_s"] + data["lat_e"]) / 2
            mid_lon = (data["lon_s"] + data["lon_e"]) / 2
            for fc_lat, fc_lon in self._flood_cells:
                dist = haversine(mid_lat, mid_lon, fc_lat, fc_lon)
                if dist < self.FLOOD_PENALTY_RADIUS_M:
                    self.graph[u][v]["weight"] *= self.FLOOD_WEIGHT_MULTIPLIER
                    self.graph[u][v]["flood_risk"] = True
                    break

    def _snap_to_node(self, lat: float, lon: float) -> tuple:
        """Find the nearest graph node to given lat/lon."""
        nodes = list(self.graph.nodes)
        if not nodes:
            raise ValueError("Empty graph")
        best = min(nodes, key=lambda n: haversine(lat, lon, n[0], n[1]))
        return best

    def compute_safe_route(
        self,
        src_lat: float, src_lon: float,
        dst_lat: float, dst_lon: float
    ) -> Dict[str, Any]:
        """Compute safest path and return GeoJSON polyline + metadata."""

        # Lazy-load flood zones from hotspot engine
        if not self._flood_cells:
            from .hotspot_engine import HotspotEngine
            he = HotspotEngine()
            self._flood_cells = he.get_high_risk_cells()
            self._apply_flood_penalties()

        src_node = self._snap_to_node(src_lat, src_lon)
        dst_node = self._snap_to_node(dst_lat, dst_lon)

        try:
            path_nodes = nx.shortest_path(self.graph, src_node, dst_node, weight="weight")
        except nx.NetworkXNoPath:
            raise ValueError("No path found between selected points")

        # Build GeoJSON coordinates [lon, lat]
        coords = [[node[1], node[0]] for node in path_nodes]

        # Calculate total distance and flood zones avoided
        total_dist_m = 0.0
        flood_edges = 0
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            if self.graph.has_edge(u, v):
                data = self.graph[u][v]
                # Use raw distance (not penalized weight) for actual distance
                segment = haversine(u[0], u[1], v[0], v[1])
                total_dist_m += segment
                if data.get("flood_risk"):
                    flood_edges += 1

        # How many high-risk zones are NEAR the route (for display)
        avoided = len(self._flood_cells) - flood_edges
        est_time_min = round((total_dist_m / 1000) / self.SPEED_KMPH * 60, 1)

        return {
            "safe_route_geojson": {
                "type": "Feature",
                "properties": {
                    "total_distance_m": round(total_dist_m, 1),
                    "estimated_time_min": est_time_min,
                    "avoided_flood_zones": max(0, avoided),
                    "route_segments": len(path_nodes) - 1,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                }
            },
            "distance_km": round(total_dist_m / 1000, 2),
            "avoided_flood_zones": max(0, avoided),
            "estimated_time_min": est_time_min,
        }
