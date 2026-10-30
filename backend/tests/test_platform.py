"""
Comprehensive Automated Software & Machine Learning Test Suite (`test_platform.py`)
Covers Section 12 & Section 16 requirements:
  1. Software & API Integration Tests (Health, Input Validation, Corrupt/Invalid File Rejection,
     Image & Video Detection API, Detection Review Persistence API, Recovery Prediction API,
     Incident Status Update & Audit Trail Persistence, CSV Export, Traffic Trends, Data Sources Status,
     Road-Network Dijkstra Detour & Shockwave API, Security Headers, PBKDF2 Auth)
  2. Machine-Learning Pipeline Tests (Data Loading, Missing-Modality Handling, Feature Consistency,
     Target-Leakage Firewall, Strict Chronological Train/Val/Test Separation, Multimodal Ablation,
     Baseline vs. Quantile GBR Regression Benchmarks, and Target Distinction).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.data_pipeline import (
    FEATURE_COLUMNS,
    LEAKAGE_FORBIDDEN_FEATURES,
    chronological_train_val_test_split,
    generate_benchmark_dataset,
    preprocess_features,
    validate_no_target_leakage,
)
from app.database import hash_password, verify_password
from app.main import app
from app.network_analysis import RoadNetworkAnalyzer


class TestMultimodalTrafficPlatform(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.media_dir = BACKEND_DIR / "data" / "media"
        cls.bundle_path = BACKEND_DIR / "data" / "platform_bundle.json"

    def test_01_health_and_security_headers(self) -> None:
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["database_connected"])
        self.assertEqual(data["road_network_segments"], 25)
        self.assertIn("inference_telemetry", data)
        self.assertEqual(resp.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(resp.headers.get("X-Frame-Options"), "DENY")

    def test_02_chronological_split_and_zero_temporal_leakage(self) -> None:
        df = generate_benchmark_dataset(num_records=200, seed=42)
        tr, va, te = chronological_train_val_test_split(df, 0.65, 0.15)
        self.assertEqual(len(tr) + len(va) + len(te), 200)
        self.assertLessEqual(tr["timestamp"].max(), va["timestamp"].min())
        self.assertLessEqual(va["timestamp"].max(), te["timestamp"].min())

    def test_03_target_leakage_firewall(self) -> None:
        for forbidden_col in LEAKAGE_FORBIDDEN_FEATURES:
            with self.assertRaises(ValueError):
                validate_no_target_leakage(["total_lanes", "speed_ratio", forbidden_col])

    def test_04_missing_modality_handling_and_feature_consistency(self) -> None:
        df = generate_benchmark_dataset(num_records=50, seed=99)
        X, cols = preprocess_features(df)
        self.assertEqual(cols, FEATURE_COLUMNS)
        self.assertEqual(X.shape, (50, len(FEATURE_COLUMNS)))
        self.assertFalse(np.any(np.isnan(X)))
        self.assertFalse(np.any(np.isinf(X)))
        vis_col_idx = FEATURE_COLUMNS.index("vis_collision_score_clean")
        self.assertTrue(np.all(X[:, vis_col_idx] >= 0.0))

    def test_05_file_upload_validation_and_rejection(self) -> None:
        resp_bad_ext = self.client.post(
            "/api/incidents/detect",
            files={"file": ("malware.exe", b"MZ\x90\x00\x03", "application/octet-stream")},
        )
        self.assertEqual(resp_bad_ext.status_code, 400)
        self.assertIn("Unsupported file format", resp_bad_ext.json()["detail"])

        resp_corrupt = self.client.post(
            "/api/incidents/detect",
            files={"file": ("corrupt.jpg", b"not_a_valid_jpeg_header_12345", "image/jpeg")},
        )
        self.assertEqual(resp_corrupt.status_code, 400)
        self.assertIn("Corrupt or invalid image", resp_corrupt.json()["detail"])

    def test_06_cv_detection_and_operator_review_persistence(self) -> None:
        collision_img = self.media_dir / "cctv_highway_collision.jpg"
        self.assertTrue(collision_img.exists(), "Seeded CCTV collision image must exist")
        resp = self.client.post(
            "/api/incidents/detect",
            files={"file": ("cctv_highway_collision.jpg", collision_img.read_bytes(), "image/jpeg")},
            data={"segment_id": "SEG-08", "persist_record": "true"},
        )
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        det_id = res["detection_id"]
        self.assertEqual(res["media_type"], "image")
        self.assertGreater(res["primary_result"]["detected_objects_count"], 0)

        # Operator review confirmation persisted in SQLite
        rev_resp = self.client.post(
            f"/api/detections/{det_id}/review",
            json={"outcome": "confirmed", "reviewed_by": "tmc_admin"},
        )
        self.assertEqual(rev_resp.status_code, 200)
        self.assertEqual(rev_resp.json()["detection"]["review_outcome"], "confirmed")
        self.assertEqual(rev_resp.json()["detection"]["verification_status"], "confirmed_incident")

        # Video clip processing
        vid_path = self.media_dir / "sample_incident_clip.avi"
        if vid_path.exists():
            resp_vid = self.client.post(
                "/api/incidents/detect",
                files={"file": ("sample_incident_clip.avi", vid_path.read_bytes(), "video/x-msvideo")},
                data={"segment_id": "SEG-08", "frame_sample_fps": "1.5"},
            )
            self.assertEqual(resp_vid.status_code, 200)
            self.assertEqual(resp_vid.json()["media_type"], "video")

    def test_07_recovery_prediction_and_incident_status_audit_persistence(self) -> None:
        payload = {
            "segment_id": "SEG-08",
            "road_class": "expressway",
            "total_lanes": 4,
            "lanes_blocked": 2,
            "is_bottleneck": 0,
            "free_flow_speed_kmh": 75.0,
            "current_speed_kmh": 22.0,
            "flow_vph": 5800,
            "capacity_vph": 7600,
            "occupancy_pct": 74.0,
            "weather_condition": "heavy_rain",
            "precipitation_mm": 18.5,
            "visibility_km": 2.4,
            "hour_of_day": 9,
            "day_of_week": 1,
            "has_visual_modality": 1,
            "vis_collision_score": 0.86,
            "vis_debris_score": 0.42,
            "vis_stopped_vehicles": 2,
            "vis_queue_density": 0.78,
            "vis_confidence": 0.91,
            "reference_timestamp": "2026-10-20T09:15:00",
            "persist_record": True,
        }
        resp = self.client.post("/api/recovery/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        clr = body["recovery_prediction"]["incident_clearance_prediction"]
        net = body["recovery_prediction"]["road_network_recovery_prediction"]

        self.assertGreater(net["predicted_minutes"], clr["predicted_minutes"])
        self.assertLess(clr["prediction_interval_80_pct"][0], clr["predicted_minutes"])
        self.assertGreater(clr["prediction_interval_80_pct"][1], clr["predicted_minutes"])

        inc_id = body["incident_id"]
        # Update incident status and verification state in SQLite
        patch_resp = self.client.patch(
            f"/api/incidents/{inc_id}/status",
            json={
                "status": "confirmed",
                "verification_status": "confirmed",
                "actor": "corridor_operator",
                "note": "Confirmed via CCTV Cam 01 and dispatched tow unit.",
            },
        )
        self.assertEqual(patch_resp.status_code, 200)
        updated_inc = patch_resp.json()["incident"]
        self.assertEqual(updated_inc["status"], "confirmed")
        self.assertEqual(updated_inc["verification_status"], "confirmed")
        self.assertGreaterEqual(len(updated_inc["status_history"]), 2)

    def test_08_input_validation_rejection(self) -> None:
        bad_payload = {
            "segment_id": "SEG-08",
            "road_class": "expressway",
            "total_lanes": 3,
            "lanes_blocked": 5,
        }
        resp = self.client.post("/api/recovery/predict", json=bad_payload)
        self.assertEqual(resp.status_code, 422)

    def test_09_road_network_shockwave_and_dijkstra_detour(self) -> None:
        analyzer = RoadNetworkAnalyzer()
        res = analyzer.analyze_incident_impact(
            segment_id="SEG-08",
            lanes_blocked=3,
            severity="critical",
            predicted_clearance_min=52.0,
            predicted_network_recovery_min=86.0,
            current_speed_kmh=14.0,
        )
        self.assertEqual(res["snapped_segment"]["segment_id"], "SEG-08")
        self.assertGreaterEqual(len(res["affected_segments"]), 2)
        routing = res["routing_and_detour_analysis"]
        self.assertTrue(routing["detour_available"])
        self.assertNotIn("SEG-08", routing["alternative_detour_route"]["segments"])

    def test_10_trends_datasources_csv_export_and_ml_benchmarks(self) -> None:
        pw_hash = hash_password("SecureTrafficPass!99")
        self.assertTrue(verify_password("SecureTrafficPass!99", pw_hash))
        self.assertFalse(verify_password("WrongPassword", pw_hash))

        trends_resp = self.client.get("/api/traffic/trends")
        self.assertEqual(trends_resp.status_code, 200)
        self.assertEqual(len(trends_resp.json()["hourly_series"]), 24)

        ds_resp = self.client.get("/api/datasources/status")
        self.assertEqual(ds_resp.status_code, 200)
        self.assertGreaterEqual(len(ds_resp.json()["datasources"]), 3)

        csv_resp = self.client.get("/api/incidents/export/csv")
        self.assertEqual(csv_resp.status_code, 200)
        self.assertIn("incident_id,timestamp", csv_resp.text)

        resp = self.client.get("/api/models/metrics")
        self.assertEqual(resp.status_code, 200)
        metrics = resp.json()
        clr_bench = metrics["target_a_incident_clearance_regression"]
        net_bench = metrics["target_b_road_network_recovery_regression"]

        self.assertLess(
            clr_bench["multimodal_quantile_gbr_selected"]["mae_minutes"],
            clr_bench["stratified_median_baseline"]["mae_minutes"],
        )
        self.assertLess(
            net_bench["multimodal_quantile_gbr_selected"]["mae_minutes"],
            net_bench["stratified_median_baseline"]["mae_minutes"],
        )


    def test_11_frontend_serving_and_ui_integrity(self) -> None:
        root_resp = self.client.get("/")
        self.assertEqual(root_resp.status_code, 200)
        html = root_resp.text
        self.assertIn("Multimodal Traffic Incident Detection", html)
        self.assertIn(".sev-critical", html)
        self.assertIn("html.light-theme", html)
        self.assertIn("t.avg_speed_kmh", html)
        self.assertIn("syncWithLiveBackend", html)

        # Verify frontend preset detection review payload (uses review_outcome)
        rev_resp = self.client.post(
            "/api/detections/1/review",
            json={"review_outcome": "confirmed", "corrected_category": None, "reviewed_by": "corridor_operator"},
        )
        self.assertEqual(rev_resp.status_code, 200)
        self.assertEqual(rev_resp.json()["detection"]["review_outcome"], "confirmed")

    def test_12_route_aliases_and_env_config(self) -> None:
        alias_csv = self.client.get("/api/incidents/export.csv")
        self.assertEqual(alias_csv.status_code, 200)
        self.assertIn("incident_id,timestamp", alias_csv.text)

        env_example = Path(__file__).resolve().parent.parent.parent / ".env.example"
        self.assertTrue(env_example.exists())
        launcher = Path(__file__).resolve().parent.parent.parent / "run_platform.py"
        self.assertTrue(launcher.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
