Multimodal Traffic Incident Detection & Road Network Recovery Time Prediction Platform (v2.2.0-OPS)

An enterprise-grade Intelligent Transportation Systems (ITS) and Traffic Management Center (TMC) operations platform modeled on the 14-node, 25-directed-segment Greater Chennai IT Corridor (OMR, Guindy, Sardar Patel Road, Velachery, Pallikaranai, and GST Road). The platform integrates OpenCV multiscale computer vision, confidence-gated multimodal late fusion, conformalized quantile gradient boosted regression for dual-target recovery forecasting, and Lighthill-Whitham-Richards (LWR) kinematic wave / Dijkstra shortest-time detour routing with a 10-view interactive Operations Center web interface.

Repository: https://github.com/JithamanyuBabu/Multimodal-Traffic-Incident-Detection-and-Road-Network-Recovery-Time-Prediction.git

1. Key Capabilities & Technical Highlights

Computer Vision CCTV & Video Pipeline (OpenCV-Multiscale-TrafficScene-v2.1): Processes static traffic camera frames (.jpg, .png, .webp) and sampled video streams (.avi, .mp4 at 1.5 FPS) using a 22% horizon Region-of-Interest (ROI) filter and Non-Maximum Suppression (NMS) bounding-box detection to extract collision proximity, cargo debris chroma, stopped vehicle counts, upstream queue density, and wet-asphalt specular reflection scores.

Confidence-Gated Multimodal Late Fusion: Combines structured loop-detector and weather telemetry with visual scene embeddings via temperature-scaled softmax calibration. Achieves 0.9615 Macro F1 on 5-class incident classification (+0.3014 Macro F1 gain over structured-only baselines) and 0.8016 Macro F1 on 4-level severity estimation, with automatic zero-imputation fallback (has_visual_modality=0) when camera feeds are unavailable.

Dual-Target Recovery Time Prediction (GBR-Quantile-v2.1): Enforces strict operational separation between Target A (incident_clearance_minutes: physical roadway scene clearance) and Target B (network_recovery_minutes: full upstream queue dissipation and restoration to $\ge 90\%$ free-flow speed). Delivers calibrated 80% conformal prediction intervals (P10–P90) evaluated on a chronological train/validation/test split (390 / 90 / 120) guarded by an automated target-leakage firewall.

Topological LWR Shockwave & Dijkstra Detour Engine: Models remaining link capacity, shockwave propagation speed (km/h), maximum upstream queue length (km), 1st/2nd-order spillback cascades across upstream corridor links, and advisory alternative detours via Dijkstra shortest-time routing.

10-View Enterprise Operations Center UI/UX: Built with an enterprise transportation visual system (#F4F7FB canvas, #FFFFFF surfaces, #172B45 collapsible navigation sidebar, #1769D2 primary blue, and one-click Dark Operations theme toggle) featuring dual Leaflet OSM / SVG Directed Graph mapping, drag-and-drop CCTV upload, human-in-the-loop detection review, sortable incident management with persisted SQLite status audit trails, and RFC-4180 CSV export.

2. Project Directory Structure

multimodal_traffic_platform/
├── .env.example                     # Environment variable template with safe local defaults
├── .gitignore                       # Git ignore rules for Python, virtualenvs, and temp uploads
├── ARCHITECTURE.md                  # Multimodal fusion, LWR shockwave, and dual-target ML design
├── DATASET_MANIFEST.md              # Dataset provenance, feature schema, and leakage firewall rules
├── MODEL_EVALUATION.md              # Chronological split metrics, ablation tables, and calibration
├── README.md                        # Complete setup, architecture, benchmark, and API guide
├── requirements.txt                 # Pinned Python dependencies (FastAPI, Uvicorn, OpenCV, NumPy, Pandas)
├── run_platform.py                  # Cross-platform one-command launcher with health probe & browser launch
├── start_platform.ps1               # Windows PowerShell one-command launcher
├── start_platform.bat               # Windows Command Prompt / double-click batch launcher
├── stop_platform.ps1                # Windows PowerShell clean Port 8000 shutdown script
├── ui_ux_image_pipeline.py          # Automated OpenCV bbox overlay generator & UI/UX feature verifier
├── backend/
│   ├── train_and_seed.py            # Offline model training, CV asset verification, & SQLite seeding
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI application serving `/` UI and `/api/*` REST endpoints
│   │   ├── database.py              # Cross-platform SQLite persistence & incident status audit engine
│   │   ├── cv_pipeline.py           # OpenCV multiscale vehicle, collision, debris, & wet-road detector
│   │   ├── data_pipeline.py         # 30-feature engineering pipeline, chronological split, & firewall
│   │   ├── multimodal_fusion.py     # Temperature-scaled confidence-gated multimodal classifier
│   │   ├── recovery_model.py        # Dual-target conformal quantile GBR + baseline regressors
│   │   └── network_analysis.py      # LWR queue shockwave & Dijkstra shortest-time detour routing
│   ├── data/
│   │   ├── incidents_multimodal_benchmark.csv  # 600-record chronological corridor benchmark dataset
│   │   ├── road_network_topology.json          # 14-node, 25-segment Greater Chennai IT Corridor graph
│   │   ├── platform_bundle.json                # Serialized trees, metrics, & analyzed CCTV metadata
│   │   ├── traffic_platform.db                 # Seeded SQLite database (incidents, detections, audit logs)
│   │   └── media/                              # Benchmark CCTV frames, annotated JPEGs, & AVI video clip
│   └── tests/
│       └── test_platform.py         # 12 automated unit, integration, security, and UI integrity tests
└── frontend/
    ├── index.html                   # Self-contained 10-view Enterprise Operations Center UI
    ├── package.json                 # Optional TypeScript / Vite frontend workspace manifest
    ├── vite.config.ts
    └── src/


3. Quick Start (Windows 11 PowerShell / CMD & Linux / macOS)

Option A — One-Command Automated Startup (Recommended)

clone the repository and run the launcher in Windows PowerShell:

git clone https://github.com/JithamanyuBabu/Multimodal-Traffic-Incident-Detection-and-Road-Network-Recovery-Time-Prediction.git
cd Multimodal-Traffic-Incident-Detection-and-Road-Network-Recovery-Time-Prediction

# Install dependencies into your active Python 3.11+ environment
python -m pip install -r requirements.txt

# Launch the full-stack platform (runs health checks and opens http://127.0.0.1:8000)
python run_platform.py


You can also start and stop the server on Windows using the included scripts:

.\start_platform.ps1
.\stop_platform.ps1


Or via Command Prompt / File Explorer:

start_platform.bat


Option B — Full Training, CV Pipeline Verification & Test Suite Execution

Use python -m with your active Python interpreter so all commands run in the exact same environment:

# 1. Train models, verify CCTV media assets, and seed SQLite + platform_bundle.json
python backend\train_and_seed.py

# 2. Generate annotated OpenCV bounding-box JPEGs and verify all UI/UX features
python ui_ux_image_pipeline.py

# 3. Run the 12-test unit, ML firewall, CV upload, and API integration suite
python -m unittest discover -s backend/tests -v

# 4. Start the FastAPI server on http://127.0.0.1:8000
python run_platform.py


4. Held-Out Chronological Evaluation Benchmarks (n = 120 Test Split)

All models are trained and evaluated using a strict chronological split (65% Train [n = 390], 15% Validation [n = 90] for temperature scaling and conformal quantile calibration, and 20% Held-Out Future Test [n = 120]) with an automated target-leakage firewall (LEAKAGE_FORBIDDEN_FEATURES).

A. Multimodal Incident Category & Severity Classification Ablation

Task & Modality Configuration

Accuracy

Macro Precision

Macro Recall

Macro F1

Weighted F1

Brier Score

Category — Structured-Only Baseline

63.3%

0.6674

0.6657

0.6601

0.6342

0.4484

Category — Visual-Only Baseline

88.3%

0.9167

0.9027

0.9046

0.8865

0.1209

Category — Early Feature Concatenation

92.5%

0.9373

0.9326

0.9344

0.9257

0.1117

Category — Confidence-Gated Late Fusion (Selected)

95.8%

0.9603

0.9630

0.9615

0.9584

0.0765

Severity — Structured-Only Baseline

78.3%

0.7873

0.7859

0.7821

0.7930

0.3188

Severity — Visual-Only Baseline

62.5%

0.6977

0.6666

0.6672

0.6508

0.4289

Severity — Early Feature Concatenation

75.8%

0.7402

0.7526

0.7444

0.7614

0.3048

Severity — Confidence-Gated Late Fusion (Selected)

80.8%

0.8059

0.8035

0.8016

0.8156

0.2999

B. Dual-Target Recovery Time Regression Benchmarks (Target A vs. Target B)

Prediction Target

Candidate Algorithm

MAE (min)

RMSE (min)

Median AE (min)

$R^2$ Score

80% Interval Coverage

Target A: Scene Clearance

Stratified Median Baseline

8.73

10.72

8.12

0.8438

N/A

Target A: Scene Clearance

Ridge Linear Regressor

5.22

6.29

4.65

0.9462

N/A

Target A: Scene Clearance

Random Forest Regressor (28 Trees)

5.78

7.57

4.35

0.9221

65.8%

Target A: Scene Clearance

Structured-Only Gradient Boosting

6.16

7.76

5.19

0.9181

N/A

Target A: Scene Clearance

Multimodal Quantile GBR (Selected)

5.06

6.50

4.63

0.9426

77.5%

Target B: Network Recovery

Stratified Median Baseline

23.11

30.40

16.20

0.7089

N/A

Target B: Network Recovery

Ridge Linear Regressor

10.96

14.04

9.22

0.9379

N/A

Target B: Network Recovery

Random Forest Regressor (28 Trees)

11.45

16.08

8.41

0.9185

75.0%

Target B: Network Recovery

Structured-Only Gradient Boosting

11.50

14.91

9.21

0.9300

N/A

Target B: Network Recovery

Multimodal Quantile GBR (Selected)

9.88

13.91

7.33

0.9390

87.5%

5. 10-View Enterprise Operations Center Architecture

Overview Workspace (data-nav="overview"): Clickable KPI summary row with drill-down navigation, corridor topological map, active priority incident queue, CCTV & computer vision quick-monitor panel, data-source health status, and 24-hour speed/flow/recovery trend chart.

Live Traffic Map (data-nav="livetraffic"): Interactive Leaflet OpenStreetMap + offline SVG Directed Graph schematic of the 14-node, 25-segment Chennai IT Corridor with full-screen mode, corridor bounds reset, congestion link styling, LWR shockwave overlays, modeled Dijkstra detour polylines, and incident inspector panel.

CCTV & Incident Detection (data-nav="detection"): Responsive 4-camera gallery (Cam 01: OMR Perungudi Collision, Cam 02: Pallikaranai Radial Debris, Cam 03: Sholinganallur Free-Flow, Cam 04: 5-Frame AVI Incident Sequence), drag-and-drop image/video upload with corrupt-file rejection, interactive HTML5 Canvas NMS bounding-box overlay, human-in-the-loop review (Confirm, Reject, Correct Class), and one-click feature piping into the Recovery Predictor.

Incident Management (data-nav="management"): Enterprise registry with multi-criteria search, status/severity/category filtering, clickable sortable column headers, pagination, RFC-4180 CSV export, and slide-over Incident Detail & Status Audit Drawer.

Recovery Prediction (data-nav="recovery"): Interactive multimodal what-if simulator for road geometry, loop-detector flow/speed/occupancy, monsoon weather, and visual modality gating (has_visual_modality), outputting Target A & Target B predictions, 80% conformal intervals, and per-instance feature attributions.

Road Network Impact (data-nav="networkimpact"): Before/after segment speed and capacity tables, LWR queue propagation metrics, and baseline vs. disrupted vs. modeled Dijkstra detour route comparisons.

Analytics & History (data-nav="analytics"): Category distribution doughnut chart, Observed vs. Predicted recovery scatter plot, MAE by severity bar chart, and historical audit log.

Model Performance (data-nav="performance"): Complete multimodal ablation tables, dual-target regression benchmarks, interactive 5×5 category and 4×4 severity confusion matrices, and top-10 permutation feature importance chart.

Data Sources (data-nav="datasources"): Dataset governance cards (DS-01 through DS-04), licensing metadata, and visual inspection of the LEAKAGE_FORBIDDEN_FEATURES firewall.

Settings & API (data-nav="settings"): Active regression engine switcher (GBR-Quantile-v2.1, Ridge-Linear-v2.1, Stratified-Median-v1.0), CV auto-confirmation threshold slider, and live FastAPI JSON endpoint inspector.

6. REST API Endpoints (http://127.0.0.1:8000)

Method

Endpoint

Description

GET

/

Serves the 10-view Enterprise Operations Center web application (frontend/index.html)

GET

/api/health

Service health check, active model versions, SQLite connectivity, and security headers

POST

/api/incidents/detect

Multipart image/video upload analyzed via OpenCV (process_media_bytes)

POST

/api/detections/{id}/review

Persist operator confirmation, false-positive rejection, or category correction

POST

/api/recovery/predict

Execute multimodal fusion, dual-target recovery prediction, & LWR/Dijkstra impact

GET

/api/incidents

Filterable incident list joined with latest recovery predictions

GET

/api/incidents/{id}

Full incident detail including persisted status_history audit trail

PATCH

/api/incidents/{id}/status

Update incident status (active, under_review, confirmed, clearing, resolved)

GET

/api/incidents/export/csv

Download filtered incident and recovery records as an RFC-4180 CSV file

POST

/api/network/analyze

Compute LWR upstream queue length, capacity drop, and Dijkstra detour route

GET

/api/network/topology

Return the 14-node, 25-segment directed road network graph definition

GET

/api/traffic/trends

Return 24-hour corridor speed, volume, incident count, and recovery trends

GET

/api/datasources/status

Return data source provenance (CACHED, UPLOADED, INFERRED, LIVE) and latency

GET

/api/models/metrics

Return chronological train/val/test evaluation metrics and ablation tables

GET

/api/datasets/manifest

Return dataset governance manifest and leakage-forbidden column firewall

7. Automated Test Suite (backend/tests/test_platform.py)

Run all 12 unit and integration tests from the repository root:

python -m unittest discover -s backend/tests -v


test_01_health_and_security_headers: Verifies /api/health, database readiness, 25 road segments, and X-Content-Type-Options / X-Frame-Options headers.

test_02_chronological_split_and_zero_temporal_leakage: Verifies strict temporal ordering ($t_{\text{train}} \le t_{\text{val}} \le t_{\text{test}}$).

test_03_target_leakage_firewall: Confirms ValueError is raised if any post-event column in LEAKAGE_FORBIDDEN_FEATURES is passed to validate_no_target_leakage().

test_04_missing_modality_handling_and_feature_consistency: Verifies NaN/Inf-free preprocessing when has_visual_modality=0.

test_05_file_upload_validation_and_rejection: Confirms HTTP 400 rejection of invalid extensions (.exe) and corrupt image headers.

test_06_cv_detection_and_operator_review_persistence: Verifies OpenCV detection on cctv_highway_collision.jpg and sample_incident_clip.avi plus SQLite persistence of operator reviews.

test_07_recovery_prediction_and_incident_status_audit_persistence: Confirms $T_{\text{network}} > T_{\text{clearance}}$, valid 80% prediction intervals, and persisted PATCH /api/incidents/{id}/status audit history.

test_08_input_validation_rejection: Confirms HTTP 422 rejection when lanes_blocked > total_lanes.

test_09_road_network_shockwave_and_dijkstra_detour: Verifies LWR upstream spillback identification and Dijkstra detour routing around blocked links.

test_10_trends_datasources_csv_export_and_ml_benchmarks: Verifies PBKDF2 password hashing, 24-hour traffic trends, CSV export, and GBR MAE superiority over baseline medians.

test_11_frontend_serving_and_ui_integrity: Verifies root HTML serving (GET /), severity/theme CSS classes, and preset detection review persistence.

test_12_route_aliases_and_env_config: Verifies /api/incidents/export.csv, .env.example, and run_platform.py.
