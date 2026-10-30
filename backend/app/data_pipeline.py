"""
Data Ingestion, Benchmark Dataset Generation, Validation, and Preprocessing Pipeline
(Modules A & C)
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Features strictly forbidden at inference time to prevent target leakage
LEAKAGE_FORBIDDEN_FEATURES = {
    "incident_clearance_minutes",
    "network_recovery_minutes",
    "t_cleared",
    "t_speed_restored",
    "actual_responder_units_dispatched",
    "tow_arrival_delay_min",
    "post_incident_max_queue_km",
    "observed_total_vehicle_delay_veh_hrs",
}

INCIDENT_CATEGORIES = [
    "collision_multi_vehicle",
    "collision_single_vehicle",
    "road_obstruction_debris",
    "disabled_stalled_vehicle",
    "non_incident_congestion",
]

SEVERITY_LEVELS = ["low", "moderate", "high", "critical"]

ROAD_CLASSES = ["expressway", "urban_arterial", "collector", "interchange_ramp"]

WEATHER_CONDITIONS = ["clear", "light_rain", "heavy_rain", "fog_haze"]


def get_road_network_definition() -> Dict[str, Any]:
    """
    Returns a realistic 14-node, 22-directed-segment urban road network topology
    modeled on the Greater Chennai IT Corridor & Southern Arterial Network
    (Kathipara Interchange, Guindy, Sardar Patel Rd, Perungudi, Tidel Park,
    Velachery, Thoraipakkam, Sholinganallur, Siruseri, GST Airport Corridor).
    """
    nodes = [
        {"node_id": "N01", "name": "Kathipara Grade Separator", "lat": 13.0067, "lon": 80.2045, "zone": "West Hub", "type": "interchange"},
        {"node_id": "N02", "name": "Guindy Halda Junction", "lat": 13.0063, "lon": 80.2206, "zone": "Central West", "type": "signalized_intersection"},
        {"node_id": "N03", "name": "IIT Madras / Sardar Patel Gate", "lat": 13.0064, "lon": 80.2425, "zone": "Central", "type": "signalized_intersection"},
        {"node_id": "N04", "name": "Madhya Kailash OMR Start", "lat": 13.0065, "lon": 80.2492, "zone": "Central East", "type": "interchange"},
        {"node_id": "N05", "name": "Tidel Park / Tharamani Jn", "lat": 12.9896, "lon": 80.2486, "zone": "OMR North", "type": "signalized_intersection"},
        {"node_id": "N06", "name": "Perungudi Toll Plaza Corridor", "lat": 12.9654, "lon": 80.2468, "zone": "OMR Mid", "type": "toll_corridor"},
        {"node_id": "N07", "name": "Thoraipakkam Radial Jn", "lat": 12.9416, "lon": 80.2361, "zone": "OMR Mid", "type": "signalized_intersection"},
        {"node_id": "N08", "name": "Sholinganallur IT Junction", "lat": 12.9010, "lon": 80.2279, "zone": "OMR South", "type": "major_junction"},
        {"node_id": "N09", "name": "Siruseri SIPCOT Expressway", "lat": 12.8350, "lon": 80.2265, "zone": "OMR Far South", "type": "expressway_terminal"},
        {"node_id": "N10", "name": "Velachery Vijayanagar Flyover", "lat": 12.9784, "lon": 80.2184, "zone": "South Central", "type": "interchange"},
        {"node_id": "N11", "name": "Pallikaranai Marsh Causeway", "lat": 12.9372, "lon": 80.2053, "zone": "South Central", "type": "causeway_bottleneck"},
        {"node_id": "N12", "name": "Medavakkam Koot Road Jn", "lat": 12.9172, "lon": 80.1923, "zone": "South West", "type": "signalized_intersection"},
        {"node_id": "N13", "name": "GST Road Airport Flyover", "lat": 12.9815, "lon": 80.1636, "zone": "Airport Corridor", "type": "expressway"},
        {"node_id": "N14", "name": "Pallavaram Radial Link", "lat": 12.9675, "lon": 80.1518, "zone": "Airport South", "type": "interchange"},
    ]

    segments = [
        {"segment_id": "SEG-01", "name": "Kathipara to Guindy Main", "u": "N01", "v": "N02", "road_class": "urban_arterial", "length_km": 1.85, "lanes": 4, "free_flow_speed_kmh": 65.0, "capacity_vph": 6400, "is_bottleneck": 0},
        {"segment_id": "SEG-02", "name": "Guindy to Kathipara Westbound", "u": "N02", "v": "N01", "road_class": "urban_arterial", "length_km": 1.85, "lanes": 4, "free_flow_speed_kmh": 65.0, "capacity_vph": 6400, "is_bottleneck": 0},
        {"segment_id": "SEG-03", "name": "Sardar Patel Rd Eastbound", "u": "N02", "v": "N03", "road_class": "urban_arterial", "length_km": 2.40, "lanes": 3, "free_flow_speed_kmh": 55.0, "capacity_vph": 4500, "is_bottleneck": 1},
        {"segment_id": "SEG-04", "name": "Sardar Patel Rd Westbound", "u": "N03", "v": "N02", "road_class": "urban_arterial", "length_km": 2.40, "lanes": 3, "free_flow_speed_kmh": 55.0, "capacity_vph": 4500, "is_bottleneck": 1},
        {"segment_id": "SEG-05", "name": "Adyar Approach to Madhya Kailash", "u": "N03", "v": "N04", "road_class": "collector", "length_km": 0.85, "lanes": 3, "free_flow_speed_kmh": 50.0, "capacity_vph": 4200, "is_bottleneck": 1},
        {"segment_id": "SEG-06", "name": "OMR Phase-1 Madhya Kailash to Tidel", "u": "N04", "v": "N05", "road_class": "expressway", "length_km": 1.95, "lanes": 4, "free_flow_speed_kmh": 70.0, "capacity_vph": 7200, "is_bottleneck": 0},
        {"segment_id": "SEG-07", "name": "OMR Phase-1 Tidel to Madhya Kailash", "u": "N05", "v": "N04", "road_class": "expressway", "length_km": 1.95, "lanes": 4, "free_flow_speed_kmh": 70.0, "capacity_vph": 7200, "is_bottleneck": 0},
        {"segment_id": "SEG-08", "name": "OMR Tidel Park to Perungudi", "u": "N05", "v": "N06", "road_class": "expressway", "length_km": 2.75, "lanes": 4, "free_flow_speed_kmh": 75.0, "capacity_vph": 7600, "is_bottleneck": 0},
        {"segment_id": "SEG-09", "name": "OMR Perungudi to Tidel Park", "u": "N06", "v": "N05", "road_class": "expressway", "length_km": 2.75, "lanes": 4, "free_flow_speed_kmh": 75.0, "capacity_vph": 7600, "is_bottleneck": 0},
        {"segment_id": "SEG-10", "name": "OMR Perungudi to Thoraipakkam", "u": "N06", "v": "N07", "road_class": "expressway", "length_km": 2.90, "lanes": 4, "free_flow_speed_kmh": 75.0, "capacity_vph": 7400, "is_bottleneck": 0},
        {"segment_id": "SEG-11", "name": "OMR Thoraipakkam to Perungudi", "u": "N07", "v": "N06", "road_class": "expressway", "length_km": 2.90, "lanes": 4, "free_flow_speed_kmh": 75.0, "capacity_vph": 7400, "is_bottleneck": 0},
        {"segment_id": "SEG-12", "name": "OMR Thoraipakkam to Sholinganallur", "u": "N07", "v": "N08", "road_class": "expressway", "length_km": 4.65, "lanes": 4, "free_flow_speed_kmh": 80.0, "capacity_vph": 7800, "is_bottleneck": 0},
        {"segment_id": "SEG-13", "name": "OMR Sholinganallur to Thoraipakkam", "u": "N08", "v": "N07", "road_class": "expressway", "length_km": 4.65, "lanes": 4, "free_flow_speed_kmh": 80.0, "capacity_vph": 7800, "is_bottleneck": 0},
        {"segment_id": "SEG-14", "name": "OMR Sholinganallur to Siruseri Elevated", "u": "N08", "v": "N09", "road_class": "expressway", "length_km": 7.40, "lanes": 4, "free_flow_speed_kmh": 85.0, "capacity_vph": 8000, "is_bottleneck": 0},
        {"segment_id": "SEG-15", "name": "Guindy to Velachery Main Rd", "u": "N02", "v": "N10", "road_class": "urban_arterial", "length_km": 3.25, "lanes": 3, "free_flow_speed_kmh": 55.0, "capacity_vph": 4600, "is_bottleneck": 0},
        {"segment_id": "SEG-16", "name": "Velachery to Tidel Park Link", "u": "N10", "v": "N05", "road_class": "collector", "length_km": 3.55, "lanes": 2, "free_flow_speed_kmh": 45.0, "capacity_vph": 3000, "is_bottleneck": 1},
        {"segment_id": "SEG-17", "name": "Tidel Park to Velachery Link", "u": "N05", "v": "N10", "road_class": "collector", "length_km": 3.55, "lanes": 2, "free_flow_speed_kmh": 45.0, "capacity_vph": 3000, "is_bottleneck": 1},
        {"segment_id": "SEG-18", "name": "Velachery to Pallikaranai Causeway", "u": "N10", "v": "N11", "road_class": "urban_arterial", "length_km": 4.80, "lanes": 3, "free_flow_speed_kmh": 55.0, "capacity_vph": 4400, "is_bottleneck": 1},
        {"segment_id": "SEG-19", "name": "Pallikaranai to Thoraipakkam 200ft Radial", "u": "N11", "v": "N07", "road_class": "urban_arterial", "length_km": 3.45, "lanes": 4, "free_flow_speed_kmh": 65.0, "capacity_vph": 6200, "is_bottleneck": 0},
        {"segment_id": "SEG-20", "name": "Thoraipakkam to Pallikaranai 200ft Radial", "u": "N07", "v": "N11", "road_class": "urban_arterial", "length_km": 3.45, "lanes": 4, "free_flow_speed_kmh": 65.0, "capacity_vph": 6200, "is_bottleneck": 0},
        {"segment_id": "SEG-21", "name": "Pallikaranai to Medavakkam Link", "u": "N11", "v": "N12", "road_class": "urban_arterial", "length_km": 2.65, "lanes": 3, "free_flow_speed_kmh": 55.0, "capacity_vph": 4500, "is_bottleneck": 0},
        {"segment_id": "SEG-22", "name": "Medavakkam to Sholinganallur Corridor", "u": "N12", "v": "N08", "road_class": "urban_arterial", "length_km": 4.25, "lanes": 3, "free_flow_speed_kmh": 60.0, "capacity_vph": 4800, "is_bottleneck": 0},
        {"segment_id": "SEG-23", "name": "Kathipara to GST Airport Expressway", "u": "N01", "v": "N13", "road_class": "expressway", "length_km": 5.20, "lanes": 4, "free_flow_speed_kmh": 80.0, "capacity_vph": 7800, "is_bottleneck": 0},
        {"segment_id": "SEG-24", "name": "GST Airport to Pallavaram Flyover", "u": "N13", "v": "N14", "road_class": "expressway", "length_km": 2.10, "lanes": 4, "free_flow_speed_kmh": 75.0, "capacity_vph": 7400, "is_bottleneck": 0},
        {"segment_id": "SEG-25", "name": "Pallavaram to Pallikaranai Radial Rd", "u": "N14", "v": "N11", "road_class": "urban_arterial", "length_km": 6.60, "lanes": 4, "free_flow_speed_kmh": 65.0, "capacity_vph": 6400, "is_bottleneck": 0},
    ]

    node_map = {n["node_id"]: n for n in nodes}
    for seg in segments:
        u_node = node_map[seg["u"]]
        v_node = node_map[seg["v"]]
        seg["start_lat"] = u_node["lat"]
        seg["start_lon"] = u_node["lon"]
        seg["end_lat"] = v_node["lat"]
        seg["end_lon"] = v_node["lon"]
        seg["mid_lat"] = round((u_node["lat"] + v_node["lat"]) / 2.0, 5)
        seg["mid_lon"] = round((u_node["lon"] + v_node["lon"]) / 2.0, 5)

    return {"nodes": nodes, "segments": segments}


def generate_benchmark_dataset(num_records: int = 600, seed: int = 42) -> pd.DataFrame:
    """
    Generates a reproducible, chronologically ordered multimodal traffic incident
    benchmark dataset combining visual feature embeddings, loop-detector telemetry,
    weather observations, road topology attributes, and dual ground-truth targets:
      1. `incident_clearance_minutes`: physical incident clearance time.
      2. `network_recovery_minutes`: full queue dissipation & speed restoration time.
    Clearly flags records as `benchmark_synthetic_calibrated` so they are never
    misrepresented as raw uncalibrated field observations.
    """
    rng = np.random.default_rng(seed)
    net = get_road_network_definition()
    segments = net["segments"]

    start_time = datetime(2026, 4, 1, 6, 0, 0)
    records: List[Dict[str, Any]] = []

    category_probs = [0.24, 0.22, 0.20, 0.20, 0.14]

    for idx in range(num_records):
        # Chronological progression over ~180 days
        delta_hours = idx * 7.2 + float(rng.uniform(-1.5, 1.5))
        ts = start_time + timedelta(hours=max(0.0, delta_hours))
        hour = ts.hour
        day_of_week = ts.weekday()
        is_peak = 1 if (7 <= hour <= 10 or 17 <= hour <= 20) and day_of_week < 5 else 0

        seg = segments[int(rng.integers(0, len(segments)))]
        cat = str(rng.choice(INCIDENT_CATEGORIES, p=category_probs))
        weather = str(rng.choice(WEATHER_CONDITIONS, p=[0.62, 0.22, 0.10, 0.06]))

        # Weather numerical measurements
        if weather == "clear":
            precip_mm = 0.0
            visibility_km = round(float(rng.uniform(7.5, 10.0)), 2)
            wet_surface = 0
        elif weather == "light_rain":
            precip_mm = round(float(rng.uniform(1.2, 8.5)), 2)
            visibility_km = round(float(rng.uniform(4.0, 7.2)), 2)
            wet_surface = 1
        elif weather == "heavy_rain":
            precip_mm = round(float(rng.uniform(12.0, 38.0)), 2)
            visibility_km = round(float(rng.uniform(1.2, 3.8)), 2)
            wet_surface = 1
        else:  # fog_haze
            precip_mm = round(float(rng.uniform(0.0, 0.8)), 2)
            visibility_km = round(float(rng.uniform(0.8, 2.8)), 2)
            wet_surface = 0

        total_lanes = int(seg["lanes"])
        if cat == "collision_multi_vehicle":
            lanes_blocked = int(min(total_lanes, rng.choice([2, 2, 3, max(1, total_lanes - 1)])))
        elif cat == "collision_single_vehicle":
            lanes_blocked = int(min(total_lanes, rng.choice([1, 1, 2])))
        elif cat == "road_obstruction_debris":
            lanes_blocked = int(min(total_lanes, rng.choice([1, 2, 2])))
        elif cat == "disabled_stalled_vehicle":
            lanes_blocked = int(min(total_lanes, rng.choice([1, 1, 1, 2])))
        else:  # non_incident_congestion
            lanes_blocked = 0

        lane_blockage_ratio = round(lanes_blocked / float(total_lanes), 3)

        # Traffic flow & speed telemetry at incident detection time
        base_demand_ratio = 0.78 if is_peak else 0.46
        if day_of_week >= 5:
            base_demand_ratio *= 0.75
        demand_ratio = float(np.clip(base_demand_ratio + rng.normal(0, 0.10), 0.20, 0.98))
        flow_vph = int(round(seg["capacity_vph"] * demand_ratio))

        ff_speed = float(seg["free_flow_speed_kmh"])
        speed_drop_factor = (
            0.38 * lane_blockage_ratio
            + (0.22 if cat == "collision_multi_vehicle" else 0.12 if cat != "non_incident_congestion" else 0.25)
            + (0.12 if weather in ("heavy_rain", "fog_haze") else 0.04 if weather == "light_rain" else 0.0)
            + (0.10 if is_peak else 0.0)
        )
        speed_ratio = float(np.clip(1.0 - speed_drop_factor + rng.normal(0, 0.06), 0.12, 0.96))
        current_speed_kmh = round(ff_speed * speed_ratio, 1)
        occupancy_pct = round(float(np.clip((1.0 - speed_ratio) * 85.0 + rng.normal(5, 4), 5.0, 96.0)), 1)

        # Visual modality features (with 8% missing-modality rate to test explicit missing-modality handling)
        has_visual = 1 if rng.random() > 0.08 else 0
        if has_visual:
            if cat == "collision_multi_vehicle":
                vis_collision_score = round(float(np.clip(rng.normal(0.84, 0.08), 0.45, 0.99)), 3)
                vis_debris_score = round(float(np.clip(rng.normal(0.56, 0.14), 0.15, 0.95)), 3)
                vis_stopped_vehicles = int(rng.integers(2, 5))
            elif cat == "collision_single_vehicle":
                vis_collision_score = round(float(np.clip(rng.normal(0.68, 0.10), 0.35, 0.94)), 3)
                vis_debris_score = round(float(np.clip(rng.normal(0.38, 0.12), 0.08, 0.78)), 3)
                vis_stopped_vehicles = int(rng.integers(1, 3))
            elif cat == "road_obstruction_debris":
                vis_collision_score = round(float(np.clip(rng.normal(0.28, 0.10), 0.05, 0.60)), 3)
                vis_debris_score = round(float(np.clip(rng.normal(0.82, 0.09), 0.45, 0.98)), 3)
                vis_stopped_vehicles = int(rng.integers(1, 3))
            elif cat == "disabled_stalled_vehicle":
                vis_collision_score = round(float(np.clip(rng.normal(0.22, 0.09), 0.04, 0.52)), 3)
                vis_debris_score = round(float(np.clip(rng.normal(0.24, 0.10), 0.04, 0.55)), 3)
                vis_stopped_vehicles = int(rng.integers(1, 2))
            else:
                vis_collision_score = round(float(np.clip(rng.normal(0.11, 0.06), 0.01, 0.35)), 3)
                vis_debris_score = round(float(np.clip(rng.normal(0.10, 0.06), 0.01, 0.32)), 3)
                vis_stopped_vehicles = 0
            vis_queue_density = round(float(np.clip(occupancy_pct / 100.0 + rng.normal(0, 0.07), 0.05, 0.98)), 3)
            vis_confidence = round(float(np.clip(0.88 - 0.12 * (1 - visibility_km / 10.0) + rng.normal(0, 0.04), 0.42, 0.98)), 3)
        else:
            vis_collision_score = -1.0
            vis_debris_score = -1.0
            vis_stopped_vehicles = -1
            vis_queue_density = -1.0
            vis_confidence = 0.0

        # Ground-truth Severity determination (based on physical impact, blockage, speed drop, road class)
        sev_raw = (
            2.2 * lane_blockage_ratio
            + (1.25 if cat == "collision_multi_vehicle" else 0.75 if cat == "collision_single_vehicle" else 0.65 if cat == "road_obstruction_debris" else 0.35 if cat == "disabled_stalled_vehicle" else 0.05)
            + 0.85 * (1.0 - speed_ratio)
            + (0.35 if seg["is_bottleneck"] else 0.0)
            + (0.30 if weather == "heavy_rain" else 0.15 if weather in ("light_rain", "fog_haze") else 0.0)
            + float(rng.normal(0, 0.18))
        )
        if sev_raw < 0.95:
            severity = "low"
        elif sev_raw < 1.55:
            severity = "moderate"
        elif sev_raw < 2.15:
            severity = "high"
        else:
            severity = "critical"

        # Ground-truth Target 1: Incident Clearance Duration (minutes)
        # Physical time to clear vehicles/debris from the roadway
        base_clearance = {
            "collision_multi_vehicle": 44.0,
            "collision_single_vehicle": 28.0,
            "road_obstruction_debris": 24.0,
            "disabled_stalled_vehicle": 18.0,
            "non_incident_congestion": 12.0,
        }[cat]
        clearance_min = (
            base_clearance
            + 16.5 * lanes_blocked
            + 12.0 * (1.0 - speed_ratio)
            + (11.0 if weather == "heavy_rain" else 5.5 if weather in ("light_rain", "fog_haze") else 0.0)
            + (6.5 if is_peak else 0.0)
            + (7.0 if seg["is_bottleneck"] else 0.0)
            + (5.5 if vis_collision_score > 0.75 else 0.0)
            + float(rng.normal(0, 4.2))
        )
        clearance_min = round(float(np.clip(clearance_min, 6.0, 165.0)), 1)

        # Ground-truth Target 2: Full Road-Network Recovery Time (minutes)
        # Includes clearance time PLUS upstream queue dissipation time (LWR shockwave discharge)
        # Queue builds while capacity is reduced, and discharges once capacity is restored
        effective_cap_ratio = max(0.08, (total_lanes - lanes_blocked) / float(total_lanes) * (0.72 if severity in ("high", "critical") else 0.85))
        excess_demand_ratio = max(0.04, demand_ratio - effective_cap_ratio * 0.85)
        shockwave_dissipation_min = (
            clearance_min * (0.35 + 1.15 * excess_demand_ratio)
            + (14.0 if seg["is_bottleneck"] else 4.0)
            + (9.5 if is_peak else 2.0)
            + (8.0 if seg["road_class"] == "expressway" and lanes_blocked >= 2 else 0.0)
            + float(rng.normal(0, 3.8))
        )
        network_recovery_min = round(float(np.clip(clearance_min + max(3.0, shockwave_dissipation_min), clearance_min + 3.0, 260.0)), 1)

        # Post-hoc fields (strictly excluded at inference to test leakage firewall)
        actual_units = int(1 + lanes_blocked + (2 if severity == "critical" else 1 if severity == "high" else 0))
        tow_delay = round(clearance_min * 0.34 + float(rng.uniform(1.0, 5.0)), 1)
        max_queue_km = round((network_recovery_min - clearance_min) * 0.085 * demand_ratio, 2)

        lat_jitter = float(rng.uniform(-0.0018, 0.0018))
        lon_jitter = float(rng.uniform(-0.0018, 0.0018))

        records.append({
            "incident_id": f"INC-2026-{idx + 1001:04d}",
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "segment_id": seg["segment_id"],
            "segment_name": seg["name"],
            "road_class": seg["road_class"],
            "lat": round(seg["mid_lat"] + lat_jitter, 5),
            "lon": round(seg["mid_lon"] + lon_jitter, 5),
            "total_lanes": total_lanes,
            "lanes_blocked": lanes_blocked,
            "lane_blockage_ratio": lane_blockage_ratio,
            "is_bottleneck": int(seg["is_bottleneck"]),
            "free_flow_speed_kmh": ff_speed,
            "current_speed_kmh": current_speed_kmh,
            "speed_ratio": round(speed_ratio, 3),
            "flow_vph": flow_vph,
            "capacity_vph": int(seg["capacity_vph"]),
            "volume_to_capacity": round(flow_vph / float(seg["capacity_vph"]), 3),
            "occupancy_pct": occupancy_pct,
            "weather_condition": weather,
            "precipitation_mm": precip_mm,
            "visibility_km": visibility_km,
            "wet_surface": wet_surface,
            "hour_of_day": hour,
            "day_of_week": day_of_week,
            "is_peak_hour": is_peak,
            "has_visual_modality": has_visual,
            "vis_collision_score": vis_collision_score,
            "vis_debris_score": vis_debris_score,
            "vis_stopped_vehicles": vis_stopped_vehicles,
            "vis_queue_density": vis_queue_density,
            "vis_confidence": vis_confidence,
            "incident_category": cat,
            "severity": severity,
            "incident_clearance_minutes": clearance_min,
            "network_recovery_minutes": network_recovery_min,
            "actual_responder_units_dispatched": actual_units,
            "tow_arrival_delay_min": tow_delay,
            "post_incident_max_queue_km": max_queue_km,
            "data_source": "benchmark_synthetic_calibrated",
        })

    return pd.DataFrame(records)


FEATURE_COLUMNS = [
    "total_lanes",
    "lanes_blocked",
    "lane_blockage_ratio",
    "is_bottleneck",
    "free_flow_speed_kmh",
    "current_speed_kmh",
    "speed_ratio",
    "flow_vph",
    "capacity_vph",
    "volume_to_capacity",
    "occupancy_pct",
    "precipitation_mm",
    "visibility_km",
    "wet_surface",
    "hour_of_day",
    "day_of_week",
    "is_peak_hour",
    "has_visual_modality",
    "vis_collision_score_clean",
    "vis_debris_score_clean",
    "vis_stopped_vehicles_clean",
    "vis_queue_density_clean",
    "vis_confidence",
    "road_class_expressway",
    "road_class_urban_arterial",
    "road_class_collector",
    "weather_clear",
    "weather_light_rain",
    "weather_heavy_rain",
    "weather_fog_haze",
]

STRUCTURED_ONLY_INDICES = [
    i for i, col in enumerate(FEATURE_COLUMNS)
    if not col.startswith("vis_") and col != "has_visual_modality"
]

VISUAL_ONLY_INDICES = [
    i for i, col in enumerate(FEATURE_COLUMNS)
    if col.startswith("vis_") or col == "has_visual_modality"
]


def validate_no_target_leakage(input_columns: List[str]) -> None:
    """Raises ValueError if any post-clearance or ground-truth target column is present in features."""
    leaked = LEAKAGE_FORBIDDEN_FEATURES.intersection(set(input_columns))
    if leaked:
        raise ValueError(f"Target leakage detected! Forbidden columns in feature set: {sorted(leaked)}")


def preprocess_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Validates, cleans, and transforms multimodal structured + visual records into
    a numerical feature matrix X while explicitly handling missing visual modalities
    via `has_visual_modality` indicator masks rather than silent magic-value substitution.
    """
    validate_no_target_leakage([c for c in df.columns if c in FEATURE_COLUMNS])
    work = df.copy()

    has_vis = (work.get("has_visual_modality", pd.Series(1, index=work.index)) > 0).astype(float)
    work["has_visual_modality"] = has_vis

    # Explicit zero-masking gated by `has_visual_modality` indicator
    work["vis_collision_score_clean"] = np.where(has_vis > 0, np.clip(work["vis_collision_score"].astype(float), 0.0, 1.0), 0.0)
    work["vis_debris_score_clean"] = np.where(has_vis > 0, np.clip(work["vis_debris_score"].astype(float), 0.0, 1.0), 0.0)
    work["vis_stopped_vehicles_clean"] = np.where(has_vis > 0, np.clip(work["vis_stopped_vehicles"].astype(float), 0.0, 10.0), 0.0)
    work["vis_queue_density_clean"] = np.where(has_vis > 0, np.clip(work["vis_queue_density"].astype(float), 0.0, 1.0), 0.0)
    work["vis_confidence"] = np.where(has_vis > 0, np.clip(work["vis_confidence"].astype(float), 0.0, 1.0), 0.0)

    # One-hot road class
    rc = work["road_class"].astype(str)
    work["road_class_expressway"] = (rc == "expressway").astype(float)
    work["road_class_urban_arterial"] = (rc == "urban_arterial").astype(float)
    work["road_class_collector"] = (rc.isin(["collector", "interchange_ramp"])).astype(float)

    # One-hot weather
    wc = work["weather_condition"].astype(str)
    work["weather_clear"] = (wc == "clear").astype(float)
    work["weather_light_rain"] = (wc == "light_rain").astype(float)
    work["weather_heavy_rain"] = (wc == "heavy_rain").astype(float)
    work["weather_fog_haze"] = (wc == "fog_haze").astype(float)

    validate_no_target_leakage(FEATURE_COLUMNS)
    X = work[FEATURE_COLUMNS].astype(float).to_numpy()
    return X, FEATURE_COLUMNS


def chronological_train_val_test_split(
    df: pd.DataFrame, train_frac: float = 0.65, val_frac: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset strictly in chronological order to prevent temporal leakage
    between related traffic states.
    """
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    n = len(ordered)
    n_train = int(n * train_frac)
    n_val = int(n * (train_frac + val_frac))
    return (
        ordered.iloc[:n_train].reset_index(drop=True),
        ordered.iloc[n_train:n_val].reset_index(drop=True),
        ordered.iloc[n_val:].reset_index(drop=True),
    )
