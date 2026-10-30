"""
Road-Network Impact and Recovery Analysis Module (Module H)
Represents the urban corridor network as a directed spatial graph (G = (V, E)),
snaps geolocated incidents to road segments via Haversine distance, computes
upstream shockwave queue propagation (1st and 2nd order incoming edges), and runs
Dijkstra shortest-time routing to identify optimal alternative detour paths.
"""
from __future__ import annotations

import heapq
import math
from typing import Any, Dict, List, Optional, Tuple

from .data_pipeline import get_road_network_definition


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


class RoadNetworkAnalyzer:
    """
    Graph-based road-network topology and disruption propagation analyzer.
    Explicitly distinguishes single-segment clearance from multi-link queue dissipation
    and labels static/modeled detour routing transparently.
    """

    def __init__(self) -> None:
        net = get_road_network_definition()
        self.nodes: Dict[str, Dict[str, Any]] = {n["node_id"]: n for n in net["nodes"]}
        self.segments: Dict[str, Dict[str, Any]] = {s["segment_id"]: s for s in net["segments"]}

        # Adjacency lists: outgoing[u] -> list of segment dicts, incoming[v] -> list of segment dicts
        self.outgoing: Dict[str, List[Dict[str, Any]]] = {nid: [] for nid in self.nodes}
        self.incoming: Dict[str, List[Dict[str, Any]]] = {nid: [] for nid in self.nodes}
        for seg in net["segments"]:
            self.outgoing[seg["u"]].append(seg)
            self.incoming[seg["v"]].append(seg)

    def snap_location_to_segment(self, lat: float, lon: float, preferred_segment_id: Optional[str] = None) -> Dict[str, Any]:
        if preferred_segment_id and preferred_segment_id in self.segments:
            seg = self.segments[preferred_segment_id]
            dist = haversine_km(lat, lon, seg["mid_lat"], seg["mid_lon"])
            return {"segment": seg, "snap_distance_km": round(dist, 3), "match_method": "explicit_segment_id"}

        best_seg = None
        best_dist = 1e9
        for seg in self.segments.values():
            d = haversine_km(lat, lon, seg["mid_lat"], seg["mid_lon"])
            if d < best_dist:
                best_dist = d
                best_seg = seg
        assert best_seg is not None
        return {"segment": best_seg, "snap_distance_km": round(best_dist, 3), "match_method": "nearest_haversine_centroid"}

    def _dijkstra_path(
        self,
        start_node: str,
        end_node: str,
        edge_travel_times_min: Dict[str, float],
        excluded_segments: Optional[set] = None,
    ) -> Optional[Dict[str, Any]]:
        excluded = excluded_segments or set()
        pq: List[Tuple[float, str, List[str], List[str], float]] = [(0.0, start_node, [start_node], [], 0.0)]
        visited: Dict[str, float] = {}

        while pq:
            cost_min, u, path_nodes, path_segs, dist_km = heapq.heappop(pq)
            if u == end_node and path_segs:
                return {
                    "nodes": path_nodes,
                    "node_names": [self.nodes[nid]["name"] for nid in path_nodes],
                    "segments": path_segs,
                    "travel_time_minutes": round(cost_min, 2),
                    "distance_km": round(dist_km, 2),
                }
            if u in visited and visited[u] <= cost_min:
                continue
            visited[u] = cost_min

            for seg in self.outgoing.get(u, []):
                sid = seg["segment_id"]
                if sid in excluded:
                    continue
                v = seg["v"]
                tt = edge_travel_times_min.get(sid, (seg["length_km"] / seg["free_flow_speed_kmh"]) * 60.0)
                new_cost = cost_min + tt
                if v not in visited or new_cost < visited[v]:
                    heapq.heappush(
                        pq,
                        (new_cost, v, path_nodes + [v], path_segs + [sid], dist_km + float(seg["length_km"])),
                    )
        return None

    def analyze_incident_impact(
        self,
        segment_id: Optional[str] = None,
        lat: float = 12.9654,
        lon: float = 80.2468,
        lanes_blocked: int = 2,
        severity: str = "high",
        predicted_clearance_min: float = 42.0,
        predicted_network_recovery_min: float = 68.0,
        current_speed_kmh: Optional[float] = None,
    ) -> Dict[str, Any]:
        snap = self.snap_location_to_segment(lat, lon, preferred_segment_id=segment_id)
        primary_seg = snap["segment"]
        sid = primary_seg["segment_id"]
        total_lanes = int(primary_seg["lanes"])
        blocked = max(0, min(total_lanes, int(lanes_blocked)))

        # Remaining capacity fraction (accounting for rubbernecking factor)
        rubberneck_factor = {"low": 0.90, "moderate": 0.80, "high": 0.68, "critical": 0.52}.get(severity, 0.75)
        if blocked >= total_lanes:
            rem_capacity_pct = 5.0
        else:
            rem_capacity_pct = round(max(8.0, ((total_lanes - blocked) / float(total_lanes)) * rubberneck_factor * 100.0), 1)

        ff_speed = float(primary_seg["free_flow_speed_kmh"])
        inc_speed = current_speed_kmh if (current_speed_kmh and current_speed_kmh > 0) else max(8.0, ff_speed * (rem_capacity_pct / 100.0) * 0.78)
        inc_speed = round(min(ff_speed, max(5.0, float(inc_speed))), 1)

        # Baseline vs Disrupted edge travel times (minutes)
        baseline_tt: Dict[str, float] = {}
        disrupted_tt: Dict[str, float] = {}
        for s_id, s in self.segments.items():
            base_t = (float(s["length_km"]) / float(s["free_flow_speed_kmh"])) * 60.0 * 1.12
            baseline_tt[s_id] = base_t
            disrupted_tt[s_id] = base_t

        # Primary segment delay
        primary_disrupted_t = (float(primary_seg["length_km"]) / max(5.0, inc_speed)) * 60.0
        disrupted_tt[sid] = primary_disrupted_t

        # Shockwave propagation to 1st-order and 2nd-order upstream segments
        queue_lag = max(4.0, predicted_network_recovery_min - predicted_clearance_min)
        shockwave_speed_kmh = round(max(4.5, 18.0 * (1.0 - rem_capacity_pct / 100.0)), 1)
        estimated_queue_length_km = round(min(9.5, (predicted_clearance_min / 60.0) * shockwave_speed_kmh * 0.65), 2)

        affected_segments: List[Dict[str, Any]] = [
            {
                "segment_id": sid,
                "name": primary_seg["name"],
                "impact_tier": "primary_incident_link",
                "u_node": primary_seg["u"],
                "v_node": primary_seg["v"],
                "road_class": primary_seg["road_class"],
                "remaining_capacity_pct": rem_capacity_pct,
                "estimated_speed_kmh": inc_speed,
                "free_flow_speed_kmh": ff_speed,
                "baseline_travel_time_min": round(baseline_tt[sid], 2),
                "disrupted_travel_time_min": round(primary_disrupted_t, 2),
                "estimated_segment_recovery_min": round(predicted_network_recovery_min, 1),
            }
        ]

        # 1st-order upstream edges (feeding into u_node of primary segment, excluding reverse direction of primary)
        u_node = primary_seg["u"]
        v_node = primary_seg["v"]
        for up1 in self.incoming.get(u_node, []):
            if up1["u"] == v_node:
                continue  # skip opposing carriageway
            up1_id = up1["segment_id"]
            spillback_factor = 1.0 + (1.0 - rem_capacity_pct / 100.0) * 0.85
            disrupted_tt[up1_id] = baseline_tt[up1_id] * spillback_factor
            up1_speed = round(float(up1["free_flow_speed_kmh"]) / spillback_factor, 1)
            up1_rec = round(predicted_clearance_min + queue_lag * 0.78, 1)
            affected_segments.append({
                "segment_id": up1_id,
                "name": up1["name"],
                "impact_tier": "1st_order_upstream_spillback",
                "u_node": up1["u"],
                "v_node": up1["v"],
                "road_class": up1["road_class"],
                "remaining_capacity_pct": round(min(92.0, rem_capacity_pct + 28.0), 1),
                "estimated_speed_kmh": up1_speed,
                "free_flow_speed_kmh": float(up1["free_flow_speed_kmh"]),
                "baseline_travel_time_min": round(baseline_tt[up1_id], 2),
                "disrupted_travel_time_min": round(disrupted_tt[up1_id], 2),
                "estimated_segment_recovery_min": up1_rec,
            })

            # 2nd-order upstream edges if queue length > up1 length
            if estimated_queue_length_km > float(up1["length_km"]) * 0.75:
                for up2 in self.incoming.get(up1["u"], []):
                    if up2["u"] == up1["v"] or up2["segment_id"] == sid:
                        continue
                    up2_id = up2["segment_id"]
                    spill2 = 1.0 + (1.0 - rem_capacity_pct / 100.0) * 0.40
                    disrupted_tt[up2_id] = baseline_tt[up2_id] * spill2
                    affected_segments.append({
                        "segment_id": up2_id,
                        "name": up2["name"],
                        "impact_tier": "2nd_order_upstream_shockwave",
                        "u_node": up2["u"],
                        "v_node": up2["v"],
                        "road_class": up2["road_class"],
                        "remaining_capacity_pct": round(min(95.0, rem_capacity_pct + 52.0), 1),
                        "estimated_speed_kmh": round(float(up2["free_flow_speed_kmh"]) / spill2, 1),
                        "free_flow_speed_kmh": float(up2["free_flow_speed_kmh"]),
                        "baseline_travel_time_min": round(baseline_tt[up2_id], 2),
                        "disrupted_travel_time_min": round(disrupted_tt[up2_id], 2),
                        "estimated_segment_recovery_min": round(predicted_clearance_min + queue_lag * 0.45, 1),
                    })

        # Compute corridor Origin-Destination route & Alternative Detour Path
        # Pick natural upstream origin and downstream destination spanning the affected segment
        origin_node = u_node
        if self.incoming.get(u_node):
            cand_up = [e["u"] for e in self.incoming[u_node] if e["u"] != v_node]
            if cand_up:
                origin_node = cand_up[0]
        dest_node = v_node
        if self.outgoing.get(v_node):
            cand_dn = [e["v"] for e in self.outgoing[v_node] if e["v"] != u_node]
            if cand_dn:
                dest_node = cand_dn[0]

        baseline_route = self._dijkstra_path(origin_node, dest_node, baseline_tt)
        disrupted_primary_route = self._dijkstra_path(origin_node, dest_node, disrupted_tt)
        detour_route = self._dijkstra_path(origin_node, dest_node, disrupted_tt, excluded_segments={sid})

        nearby_intersections = [
            self.nodes[u_node],
            self.nodes[v_node],
        ]

        return {
            "analysis_mode": "topological_lwr_shockwave_and_dijkstra_simulation",
            "snapped_segment": {
                "segment_id": sid,
                "name": primary_seg["name"],
                "snap_distance_km": snap["snap_distance_km"],
                "match_method": snap["match_method"],
            },
            "shockwave_metrics": {
                "remaining_capacity_pct": rem_capacity_pct,
                "shockwave_propagation_speed_kmh": shockwave_speed_kmh,
                "estimated_max_upstream_queue_km": estimated_queue_length_km,
                "affected_segments_count": len(affected_segments),
                "primary_clearance_min": round(predicted_clearance_min, 1),
                "full_corridor_recovery_min": round(predicted_network_recovery_min, 1),
            },
            "nearby_intersections": nearby_intersections,
            "affected_segments": affected_segments,
            "routing_and_detour_analysis": {
                "origin_node": self.nodes[origin_node],
                "destination_node": self.nodes[dest_node],
                "baseline_route": baseline_route,
                "incident_impacted_route": disrupted_primary_route,
                "alternative_detour_route": detour_route,
                "detour_available": detour_route is not None,
                "time_saved_by_detour_min": (
                    round(disrupted_primary_route["travel_time_minutes"] - detour_route["travel_time_minutes"], 2)
                    if (disrupted_primary_route and detour_route)
                    else None
                ),
            },
            "methodology_and_limitations": (
                "Road-network impact is computed on a 14-node, 25-directed-segment topological graph using "
                "Lighthill-Whitham-Richards (LWR) capacity-reduction queue propagation and Dijkstra shortest-time "
                "routing. Clearing the primary incident does not immediately restore upstream links; queue discharge "
                "continues until t = network_recovery_minutes. Detour travel times reflect modeled link states rather "
                "than live municipal loop-detector feeds unless connected to an active traffic management stream."
            ),
        }
