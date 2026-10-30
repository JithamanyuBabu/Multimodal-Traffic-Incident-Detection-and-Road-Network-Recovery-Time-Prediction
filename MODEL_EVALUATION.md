# Model Evaluation & Multimodal Ablation Report (`MODEL_EVALUATION.md`)

## 1. Validation Methodology
All machine-learning models were evaluated using a **strict chronological split** of the 600-record benchmark dataset (`backend/data/incidents_multimodal_benchmark.csv`):
- **Training Split (65%, $n=390$)**: Used for parameter estimation, tree construction, and inverse-frequency class weighting.
- **Validation Split (15%, $n=90$)**: Used strictly for post-hoc softmax temperature calibration and conformalized quantile interval padding ($\alpha \in [0.10, 0.90]$).
- **Held-Out Chronological Test Split (20%, $n=120$)**: Strictly future records ($t_{\text{test}} \ge t_{\text{val}} \ge t_{\text{train}}$) never seen during training or hyperparameter tuning.

## 2. Multimodal Fusion Ablation Study (Classification Tasks)

### 2.1 Incident Category Classification (5 Classes, Held-Out Test $n=120$)
| Modality / Fusion Strategy | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Brier Calibration Score (Lower is Better) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Structured-Only Baseline** (Loop + Weather + Road) | 0.6333 | 0.6674 | 0.6657 | 0.6601 | 0.6342 | 0.4484 |
| **Visual-Only Baseline** (OpenCV Scene Embeddings) | 0.8833 | 0.9167 | 0.9027 | 0.9046 | 0.8865 | 0.1209 |
| **Early Feature-Level Fusion** (Concatenated + Mask) | 0.9250 | 0.9373 | 0.9326 | 0.9344 | 0.9257 | 0.1117 |
| **Confidence-Gated Late Fusion (Selected)** | **0.9583** | **0.9603** | **0.9630** | **0.9615** | **0.9584** | **0.0765** |

- **Multimodal Gain over Structured-Only Baseline**: `+0.3014` Macro-F1 improvement (`+30.14%` absolute Macro-F1 gain), demonstrating that computer-vision collision and debris cues disambiguate single-lane obstructions from stalled vehicles that exhibit similar loop-detector speed drops.

### 2.2 Incident Severity Classification (4 Classes: Low, Moderate, High, Critical)
| Modality / Fusion Strategy | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Brier Calibration Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Structured-Only Baseline** | 0.7833 | 0.7873 | 0.7859 | 0.7821 | 0.7930 | 0.3188 |
| **Visual-Only Baseline** | 0.6250 | 0.6977 | 0.6666 | 0.6672 | 0.6508 | 0.4289 |
| **Early Feature-Level Fusion** | 0.7583 | 0.7402 | 0.7526 | 0.7444 | 0.7614 | 0.3048 |
| **Confidence-Gated Late Fusion (Selected)** | **0.8083** | **0.8059** | **0.8035** | **0.8016** | **0.8156** | **0.2999** |

## 3. Dual-Target Recovery-Time Prediction Benchmarks (Held-Out Test $n=120$)

### 3.1 Target A: Physical Incident Clearance Duration (`incident_clearance_minutes`)
| Candidate Model | MAE (min) | RMSE (min) | Median AE (min) | $R^2$ Score | 80% Interval Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Category-Stratified Median Baseline** | 8.73 | 10.72 | 8.12 | 0.8438 | N/A |
| **Ridge Linear Regressor** | 5.22 | 6.29 | 4.65 | 0.9462 | N/A |
| **Random Forest Regressor** (28 CART Trees) | 5.78 | 7.57 | 4.35 | 0.9221 | 65.8% |
| **Structured-Only Gradient Boosting** | 6.16 | 7.76 | 5.19 | 0.9181 | N/A |
| **Multimodal Quantile GBR (Selected)** | **5.06** | **6.50** | **4.63** | **0.9426** | **77.5%** |

### 3.2 Target B: Full Road-Network Restoration Duration (`network_recovery_minutes`)
| Candidate Model | MAE (min) | RMSE (min) | Median AE (min) | $R^2$ Score | 80% Interval Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Category-Stratified Median Baseline** | 23.11 | 30.40 | 16.20 | 0.7089 | N/A |
| **Ridge Linear Regressor** | 10.96 | 14.04 | 9.22 | 0.9379 | N/A |
| **Random Forest Regressor** (28 CART Trees) | 11.45 | 16.08 | 8.41 | 0.9185 | 75.0% |
| **Structured-Only Gradient Boosting** | 11.50 | 14.91 | 9.21 | 0.9300 | N/A |
| **Multimodal Quantile GBR (Selected)** | **9.88** | **13.91** | **7.33** | **0.9390** | **87.5%** |

## 4. Real Computer Vision Asset Verification
| Asset Filename | Media Type | Frames Analyzed | Predicted Visual Category | Confidence | Detected Objects / Boxes |
| :--- | :---: | :---: | :--- | :---: | :---: |
| `cctv_highway_collision.jpg` | Image (`JPEG`) | 1 | `collision_multi_vehicle` | `0.946` | 12 (includes `collision_vehicle_A`, `collision_vehicle_B`) |
| `cctv_arterial_debris_truck.jpg` | Image (`JPEG`) | 1 | `road_obstruction_debris` | `0.936` | 12 (includes `stalled_box_truck`, `spilled_cargo_debris`) |
| `cctv_expressway_normal_flow.jpg` | Image (`JPEG`) | 1 | `normal_free_flow` | `0.807` | 14 (well-spaced free-flowing vehicles) |
| `sample_incident_clip.avi` | Video (`AVI`) | 5 (sampled @ 1.5 FPS) | `collision_multi_vehicle` | `0.946` | 12 per sampled frame |
