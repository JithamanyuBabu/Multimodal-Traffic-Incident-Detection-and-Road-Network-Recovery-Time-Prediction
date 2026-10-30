# Dataset Manifest & Target Governance (`DATASET_MANIFEST.md`)

## 1. Executive Summary & Environment Constraints
This manifest documents all datasets, schemas, licensing constraints, target definitions, missing-modality protocols, and target-leakage safeguards used in the **Multimodal Traffic Incident Detection and Road Network Recovery Time Prediction Platform**.

Because outbound internet access is disabled inside the sandbox VM environment, external multi-gigabyte datasets (such as **US-Accidents**, **CADP**, **DAD**, and **Caltrans PeMS**) cannot be fetched over the network at runtime. Rather than inventing unverified historical claims, the platform:
1. Documents the exact schemas, licensing, and target limitations of the reference public datasets below.
2. Provides a deterministic, reproducible **600-record Multimodal Traffic Incident & Network Benchmark (`backend/data/incidents_multimodal_benchmark.csv`)** generated via `backend/app/data_pipeline.py` (`seed=42`) across a 14-node, 25-directed-segment Greater Chennai urban corridor graph (`backend/data/road_network_topology.json`), paired with 4 real OpenCV-processed CCTV image and video assets in `backend/data/media/`.
3. Clearly labels every benchmark record with `data_source = "benchmark_synthetic_calibrated"` and distinguishes them from `user_interactive_prediction` records in the SQLite database and UI.

| Dataset ID | Dataset Name & Source | License & Usage | Format | Records / Split | Target Labels Available | Key Limitations & Missing Info |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DS-01** | **Chennai Corridor Multimodal Traffic Incident & Recovery Benchmark** (`data_pipeline.py`) | MIT / Open Academic & Demo Use | CSV, SQLite, JPEG, AVI | 600 total (390 Train, 90 Val, 120 Test — strictly chronological) | `incident_category`, `severity`, `incident_clearance_minutes`, `network_recovery_minutes` | 8.0% of records intentionally omit visual modality (`has_visual_modality=0`) to validate missing-modality gating. Durations follow physical LWR queue dissipation equations; retrain on local DOT/TMC logs before field dispatch. |
| **DS-02** | **US-Accidents** (Moosavi et al., OSU / Kaggle — Reference Schema) | CC BY-NC-SA 4.0 (Non-commercial research) | Structured CSV | Reference schema | `Severity` (1–4), `Start_Time`, `End_Time` (Clearance duration) | Contains structured weather/POI/clearance timestamps, but lacks CCTV camera frames and does not measure post-clearance upstream queue dissipation time. |
| **DS-03** | **CADP & DAD** (CCTV / Dashcam Accident Datasets — Reference Schema) | Academic Research Only | MP4 / JPEG + XML Bounding Boxes | Reference schema | Frame-level accident onset, vehicle bounding boxes | Supports visual collision detection (Module B), but contains **zero** incident clearance or road-network recovery timestamps. |
| **DS-04** | **Caltrans PeMS / METR-LA** (Loop Detector Telemetry — Reference Schema) | Public Government Open Data | HDF5 / CSV Time-Series | Reference schema | 5-min speed, flow, occupancy; speed recovery timestamp ($v \ge 0.90 v_{\text{free}}$) | Provides link speed recovery curves needed for Target B (`network_recovery_minutes`), but requires spatial-temporal joining with incident dispatch logs. |

## 2. Strict Distinction Between Prediction Targets
A core scientific requirement of this platform is **never conflating visual incident detection, physical scene clearance duration, and full road-network recovery time**:

1. **Incident Detection & Classification (`incident_category`)**:
   - Identifies whether visual and/or sensor anomalies correspond to a `collision_multi_vehicle`, `collision_single_vehicle`, `road_obstruction_debris`, `disabled_stalled_vehicle`, or `non_incident_congestion`.
   - Visual anomaly detection from 2D CCTV frames is reported as a *potential incident* with calibrated confidence and bounding boxes—not an authoritative field confirmation.
2. **Target A — Incident Clearance Duration (`incident_clearance_minutes`)**:
   - Defined as $T_{\text{clearance}} = t_{\text{cleared}} - t_{\text{reported}}$, the elapsed time (in minutes) until emergency responders or tow operators remove the physical obstruction and reopen blocked lanes.
3. **Target B — Road-Network Recovery Time (`network_recovery_minutes`)**:
   - Defined as $T_{\text{network\_recovery}} = t_{\text{speed\_restored}} - t_{\text{reported}}$, the elapsed time (in minutes) until the upstream queuing shockwave ($1^{\text{st}}$-order and $2^{\text{nd}}$-order incoming segments) fully discharges and corridor traffic speeds return to $\ge 90\%$ of baseline free-flow speed ($v \ge 0.90 v_{\text{free}}$).
   - Physically and mathematically, $T_{\text{network\_recovery}} = T_{\text{clearance}} + \Delta T_{\text{queue\_dissipation}}$, where $\Delta T_{\text{queue\_dissipation}} > 0$ depends on volume-to-capacity ratio ($V/C$), lane blockage ratio, bottleneck geometry, and upstream network degree.

## 3. Target-Leakage Firewall (`LEAKAGE_FORBIDDEN_FEATURES`)
To prevent target leakage during model training and inference, `backend/app/data_pipeline.py` enforces an automated schema firewall (`validate_no_target_leakage`) that immediately raises a `ValueError` if any post-hoc variable known only *after* incident clearance is passed into the feature pipeline:
- `incident_clearance_minutes`
- `network_recovery_minutes`
- `t_cleared`
- `t_speed_restored`
- `actual_responder_units_dispatched`
- `tow_arrival_delay_min`
- `post_incident_max_queue_km`
- `observed_total_vehicle_delay_veh_hrs`
