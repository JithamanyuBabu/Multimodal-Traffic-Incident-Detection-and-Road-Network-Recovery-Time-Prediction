# Multimodal Traffic Incident Detection & Road Network Recovery Time Prediction Platform (`v2.2.0`)

An end-to-end, production-grade intelligent transportation systems (ITS) platform combining **OpenCV multiscale computer vision**, **confidence-gated multimodal late fusion**, **conformalized quantile gradient boosted regression** for dual-target recovery forecasting (**Target A: Scene Clearance Duration** vs. **Target B: Full Road-Network Restoration Duration**), and **LWR kinematic wave / Dijkstra road-network detour analysis** across the Greater Chennai IT Corridor (14 intersections, 25 directed links).

---

## 1. Complete Project Structure

```text
multimodal_traffic_platform/
├── .env.example                     # Documented environment variable template with safe defaults
├── README.md                        # Setup, architecture, API, and troubleshooting documentation
├── ARCHITECTURE.md                  # Multimodal fusion, LWR shockwave, and dual-target ML design
├── DATASET_MANIFEST.md              # Provenance, feature definitions, and leakage firewall rules
├── MODEL_EVALUATION.md              # Chronological split metrics, ablation tables, and calibration
├── requirements.txt                 # Pinned Python dependencies (FastAPI, Uvicorn, OpenCV, NumPy, Pandas)
├── run_platform.py                  # Cross-platform one-command launcher with health check & browser open
├── start_platform.ps1               # Windows PowerShell one-command launcher
├── start_platform.bat               # Windows Command Prompt / double-click batch launcher
├── stop_platform.ps1                # Windows PowerShell clean port 8000 shutdown script
├── backend/
│   ├── train_and_seed.py            # Offline training, chronological evaluation, & SQLite seeding
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI server serving both `/` web UI and `/api/*` endpoints
│   │   ├── database.py              # Cross-platform SQLite persistence & audit history engine
│   │   ├── cv_pipeline.py           # OpenCV multiscale vehicle/collision/debris/wet-asphalt detector
│   │   ├── data_pipeline.py         # Feature engineering, chronological split, & leakage firewall
│   │   ├── multimodal_fusion.py     # Temperature-scaled confidence-gated multimodal classifier
│   │   ├── recovery_model.py        # Dual-target conformal quantile GBR + baseline benchmarks
│   │   ├── network_analysis.py      # LWR queue shockwave & Dijkstra alternative detour routing
│   │   └── security.py              # PBKDF2-HMAC-SHA256 credential hashing & upload sanitizer
│   ├── data/
│   │   ├── traffic_platform.db      # Seeded SQLite database (segments, incidents, detections, models)
│   │   ├── platform_bundle.json     # Exported model weights, held-out metrics, & sample CCTV frames
│   │   └── sample_media/            # Benchmark highway collision, debris, free-flow, & video assets
│   └── tests/
│       └── test_platform.py         # 12 end-to-end unit, integration, security, and UI tests
└── frontend/
    ├── index.html                   # 10-view interactive Operations Center UI (live API + standalone)
    ├── package.json                 # Optional Vite/React TypeScript workspace manifest
    ├── vite.config.ts
    └── src/
```

---

## 2. Quick Start (Windows 11 PowerShell / CMD & Linux / macOS)

### Option A — One-Command Automated Startup (Recommended)

From the `multimodal_traffic_platform` root directory in **Windows PowerShell**:

```powershell
.\start_platform.ps1
```

Or in **Command Prompt (CMD)** / File Explorer:

```bat
start_platform.bat
```

Or on **any OS (Windows / Linux / macOS)** via Python directly:

```bash
python run_platform.py
```

**What `run_platform.py` does automatically:**
1. Verifies that `fastapi`, `uvicorn`, `opencv-python` (`cv2`), `numpy`, `pandas`, and `pydantic` are installed in the active Python environment.
2. Verifies that `backend/data/traffic_platform.db` and `backend/data/platform_bundle.json` exist (automatically running `backend/train_and_seed.py` if missing).
3. Checks whether Port `8000` is already running a healthy instance of the platform (reusing it cleanly instead of failing with a port conflict).
4. Starts the FastAPI / Uvicorn server on `http://127.0.0.1:8000`.
5. Polls `http://127.0.0.1:8000/api/health` until HTTP `200 OK` is confirmed, then opens your default web browser once at `http://127.0.0.1:8000`.

### Option B — Step-by-Step Manual Commands (Windows PowerShell)

Use the **same Python interpreter** (`python -m ...`) for installation, training, testing, and serving so packages are never installed into a mismatched Python environment:

```powershell
# 1. Navigate to the project root directory
cd multimodal_traffic_platform

# 2. Install dependencies into your active Python environment
python -m pip install -r requirements.txt

# 3. Run model training, CV verification, and SQLite database seeding
python backend\train_and_seed.py

# 4. Run the 12-test automated verification suite (from project root)
python -m unittest discover -s backend/tests -v

#   (Or if your terminal is already inside the `backend` folder:)
#   python -m unittest discover -s tests -v

# 5. Start the FastAPI server and open http://127.0.0.1:8000
python run_platform.py
```

### Stopping the Server on Windows

Press `Ctrl+C` in the terminal running Uvicorn, or run:

```powershell
.\stop_platform.ps1
```

---

## 3. REST API Endpoints (`http://127.0.0.1:8000`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the 10-view Operations Center web application (`frontend/index.html`) |
| `GET` | `/api/health` | Service health check, model versions, SQLite status, and inference telemetry |
| `POST` | `/api/incidents/detect` | Multipart image/video upload analyzed via OpenCV (`process_media_bytes`) |
| `POST` | `/api/detections/{id}/review` | Persist operator confirmation, rejection, or category correction |
| `POST` | `/api/recovery/predict` | Run multimodal fusion, dual-target recovery prediction, & LWR/Dijkstra impact |
| `GET` | `/api/incidents` | Filterable incident list joined with latest recovery predictions |
| `GET` | `/api/incidents/{id}` | Full incident detail including persisted `status_history` audit trail |
| `PATCH` | `/api/incidents/{id}/status` | Update incident status (`active`, `clearing`, `resolved`) & verification state |
| `GET` | `/api/incidents/export/csv` | Download filtered incident & recovery records as a RFC-4180 CSV file |
| `POST` | `/api/network/analyze` | Compute LWR upstream queue length, capacity drop, and Dijkstra detour route |
| `GET` | `/api/network/topology` | Return the 14-node, 25-segment directed road network graph definition |
| `GET` | `/api/traffic/trends` | Return 24-hour corridor speed, volume, incident count, and recovery trends |
| `GET` | `/api/datasources/status` | Return data source provenance (`CACHED`, `UPLOADED`, `INFERRED`) and latency |
| `GET` | `/api/models/metrics` | Return chronological train/val/test evaluation metrics and ablation tables |
| `GET` | `/api/datasets/manifest` | Return dataset governance manifest and leakage-forbidden column firewall |
