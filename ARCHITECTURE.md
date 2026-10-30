# System Architecture & Module Specification (`ARCHITECTURE.md`)

## 1. Architectural Pattern: Modular Monolith
The platform is engineered as a **Modular Monolith** combining a Python **FastAPI** backend, a persistent **SQLite** relational database (with PostgreSQL production readiness), an **OpenCV + NumPy/SciPy** machine learning and graph analysis core, and a responsive **React/TypeScript + Interactive Canvas** frontend.

```
Input Sources (CCTV Image/Video, Loop Telemetry, Weather, Road Graph, Incident Metadata)
   │
   ▼
[Module A & C: Data Ingestion, Validation & Leakage Firewall] (backend/app/data_pipeline.py)
   ├──▶ [Module B: OpenCV Multi-Scale Visual Detection Pipeline] (backend/app/cv_pipeline.py)
   │       (Frame Stride Sampling, Asphalt Background Contrast, Sobel/Canny NMS, Debris/Collision Cues)
   ▼
[Module D: Confidence-Gated Multimodal Fusion] (backend/app/multimodal_fusion.py)
   ├──▶ [Module E: Incident Category Classifier (5 Classes + Shannon Entropy Uncertainty)]
   ├──▶ [Module F: Calibrated Operational Severity Estimator (Low / Moderate / High / Critical)]
   ▼
[Module G: Dual-Target Conformalized Quantile GBR Engine] (backend/app/recovery_model.py)
   ├──▶ Target A: Incident Clearance Duration (Minutes + 80% P10–P90 Interval)
   ├──▶ Target B: Road-Network Recovery Time (Minutes + Queue Lag + 80% P10–P90 Interval)
   ▼
[Module H: Directed Graph Road-Network & LWR Shockwave Analyzer] (backend/app/network_analysis.py)
   ├──▶ Haversine Segment Snapping (14 Nodes, 25 Directed Corridor Segments)
   ├──▶ 1st & 2nd Order Upstream Spillback Propagation
   ├──▶ Dijkstra Shortest-Time Alternative Detour Path Calculation
   ▼
[Module I: Persistent SQLite Storage, FastAPI REST Layer & Interactive Web Dashboard]
   (backend/app/database.py, backend/app/main.py, frontend/)
```

## 2. Module Responsibilities
- **Module A (`data_pipeline.py`)**: Validates structured traffic, weather, and road inputs; enforces the target-leakage firewall; manages chronological 65/15/20 train/validation/test splitting.
- **Module B (`cv_pipeline.py`)**: Validates uploaded media magic bytes, file extensions (`.jpg`, `.png`, `.webp`, `.mp4`, `.avi`), file size ($\le 25\text{ MB}$), and video duration ($\le 60\text{ s}$). Extracts frames at a configurable FPS stride without loading full uncompressed videos into RAM. Computes vehicle bounding boxes with Non-Maximum Suppression (NMS), transverse collision anomaly scores, cargo/debris chroma density, wet-surface reflection indices, and upstream queue occupancy.
- **Module C (`data_pipeline.py`)**: Standardizes numerical features, encodes categorical road classes and weather states, and applies explicit missing-modality gating (`has_visual_modality` binary indicator) so missing camera feeds are never silently imputed with arbitrary constants.
- **Module D, E, F (`multimodal_fusion.py`)**: Implements `CalibratedMultinomialClassifier` (L2-regularized multinomial logistic regression with inverse-frequency class-balance weights and validation temperature scaling) across three fusion paradigms: Single-Modality Baselines, Early Feature Concatenation, and **Confidence-Gated Late Multimodal Fusion**.
- **Module G (`recovery_model.py`)**: Implements `StratifiedMedianBaseline`, `RidgeRegressionBaseline`, `RandomForestRegressorModel`, and `QuantileGradientBoostingRegressor` (`GBR-Quantile-DualTarget-v2.1`) with conformalized validation residual calibration for 80% prediction intervals (`P10`–`P90`) and instance-level factor attribution.
- **Module H (`network_analysis.py`)**: Models the 14-node, 25-segment Greater Chennai corridor as a directed spatial graph $G=(V,E)$. Snaps incident coordinates to directed links, computes remaining capacity and LWR shockwave propagation across $1^{\text{st}}$- and $2^{\text{nd}}$-order upstream segments, and runs Dijkstra shortest-time routing to compare disrupted primary routes against optimal detour paths.
- **Module I (`database.py`, `main.py`, `frontend/`)**: Persists incidents, detections, predictions, road segments, dataset manifests, model versions, and evaluation runs in SQLite; exposes validated REST endpoints with sliding-window rate limiting and security headers; renders the interactive traffic intelligence dashboard.
