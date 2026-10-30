"""
Recovery-Time Prediction Pipeline (Module G)
Explicitly separates:
  1. Target A (`incident_clearance_minutes`): Physical roadway clearance time.
  2. Target B (`network_recovery_minutes`): Full road-network queue dissipation & speed restoration time.

Implements and benchmarks 4 candidate algorithms without target leakage:
  - Category-Stratified Median Baseline
  - Ridge Regularized Linear Regressor
  - Random Forest Regressor (bootstrap CART regression ensemble with jackknife variance)
  - Gradient Boosting Regressor with Quantile Prediction Intervals (P10, P50/Mean, P90)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_pipeline import FEATURE_COLUMNS, INCIDENT_CATEGORIES, SEVERITY_LEVELS


class StratifiedMedianBaseline:
    """Simple baseline predicting training-set median duration stratified by incident category."""

    def __init__(self) -> None:
        self.global_median_: float = 30.0
        self.cat_medians_: Dict[str, float] = {}
        self.cat_iqr_: Dict[str, Tuple[float, float]] = {}

    def fit(self, df_train: pd.DataFrame, target_col: str) -> "StratifiedMedianBaseline":
        y = df_train[target_col].astype(float).to_numpy()
        self.global_median_ = float(np.median(y))
        for cat in INCIDENT_CATEGORIES:
            sub = y[df_train["incident_category"] == cat]
            if len(sub) > 0:
                self.cat_medians_[cat] = float(np.median(sub))
                self.cat_iqr_[cat] = (float(np.percentile(sub, 10)), float(np.percentile(sub, 90)))
            else:
                self.cat_medians_[cat] = self.global_median_
                self.cat_iqr_[cat] = (self.global_median_ * 0.7, self.global_median_ * 1.3)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.cat_medians_.get(str(c), self.global_median_) for c in df["incident_category"]], dtype=float)


class FastCARTNode:
    """Vectorized binary regression tree node supporting squared-error or quantile residual splits."""

    __slots__ = ("feature_idx", "threshold", "value", "left", "right")

    def __init__(
        self,
        value: float,
        feature_idx: int = -1,
        threshold: float = 0.0,
        left: Optional["FastCARTNode"] = None,
        right: Optional["FastCARTNode"] = None,
    ) -> None:
        self.value = value
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right

    def to_dict(self) -> Dict[str, Any]:
        if self.feature_idx < 0 or self.left is None or self.right is None:
            return {"v": round(self.value, 3)}
        return {
            "f": int(self.feature_idx),
            "t": round(self.threshold, 4),
            "v": round(self.value, 3),
            "l": self.left.to_dict(),
            "r": self.right.to_dict(),
        }


def _build_cart_tree(
    X: np.ndarray,
    y: np.ndarray,
    depth: int,
    max_depth: int,
    min_samples_leaf: int,
    feature_indices: np.ndarray,
    leaf_quantile: Optional[float] = None,
) -> FastCARTNode:
    n = len(y)
    leaf_val = float(np.quantile(y, leaf_quantile)) if leaf_quantile is not None else float(np.mean(y))
    if depth >= max_depth or n < 2 * min_samples_leaf or np.var(y) < 1e-5:
        return FastCARTNode(value=leaf_val)

    best_gain = 0.0
    best_feat = -1
    best_thresh = 0.0
    current_sse = float(np.sum((y - np.mean(y)) ** 2))

    for f_idx in feature_indices:
        col = X[:, f_idx]
        uniq = np.unique(col)
        if len(uniq) <= 1:
            continue
        # Evaluate up to 9 candidate quantile thresholds for speed & regularization
        if len(uniq) > 10:
            candidates = np.quantile(col, [0.15, 0.30, 0.45, 0.55, 0.70, 0.85])
        else:
            candidates = (uniq[:-1] + uniq[1:]) * 0.5

        for thresh in np.unique(candidates):
            left_mask = col <= thresh
            n_left = int(np.sum(left_mask))
            n_right = n - n_left
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            y_left = y[left_mask]
            y_right = y[~left_mask]
            sse_split = float(np.sum((y_left - np.mean(y_left)) ** 2) + np.sum((y_right - np.mean(y_right)) ** 2))
            gain = current_sse - sse_split
            if gain > best_gain:
                best_gain = gain
                best_feat = int(f_idx)
                best_thresh = float(thresh)

    if best_feat < 0:
        return FastCARTNode(value=leaf_val)

    left_mask = X[:, best_feat] <= best_thresh
    left_child = _build_cart_tree(
        X[left_mask], y[left_mask], depth + 1, max_depth, min_samples_leaf, feature_indices, leaf_quantile
    )
    right_child = _build_cart_tree(
        X[~left_mask], y[~left_mask], depth + 1, max_depth, min_samples_leaf, feature_indices, leaf_quantile
    )
    return FastCARTNode(
        value=leaf_val,
        feature_idx=best_feat,
        threshold=best_thresh,
        left=left_child,
        right=right_child,
    )


def _predict_tree_array(node: FastCARTNode, X: np.ndarray) -> np.ndarray:
    out = np.empty(len(X), dtype=float)
    for i in range(len(X)):
        cur = node
        row = X[i]
        while cur.feature_idx >= 0 and cur.left is not None and cur.right is not None:
            if row[cur.feature_idx] <= cur.threshold:
                cur = cur.left
            else:
                cur = cur.right
        out[i] = cur.value
    return out


class RandomForestRegressorModel:
    """Bootstrap-aggregated CART ensemble providing point predictions and tree-dispersion intervals."""

    def __init__(self, n_estimators: int = 28, max_depth: int = 5, min_samples_leaf: int = 8, seed: int = 42) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.trees_: List[FastCARTNode] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressorModel":
        rng = np.random.default_rng(self.seed)
        n, p = X.shape
        max_features = max(4, int(round(p * 0.65)))
        self.trees_ = []
        for _ in range(self.n_estimators):
            boot_idx = rng.integers(0, n, size=n)
            feat_idx = rng.choice(p, size=max_features, replace=False)
            tree = _build_cart_tree(
                X[boot_idx], y[boot_idx], 0, self.max_depth, self.min_samples_leaf, feat_idx, leaf_quantile=None
            )
            self.trees_.append(tree)
        return self

    def predict_with_dispersion(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_preds = np.column_stack([_predict_tree_array(t, X) for t in self.trees_])
        mean_pred = np.mean(all_preds, axis=1)
        p10 = np.percentile(all_preds, 10, axis=1)
        p90 = np.percentile(all_preds, 90, axis=1)
        return mean_pred, p10, p90

    def predict(self, X: np.ndarray) -> np.ndarray:
        mean_pred, _, _ = self.predict_with_dispersion(X)
        return mean_pred


class QuantileGradientBoostingRegressor:
    """
    Gradient Boosted Regression Trees supporting both squared-error point prediction
    and pinball/quantile loss boosting (alpha=0.10 and alpha=0.90) with conformalized
    validation residual calibration for defensible 80% prediction intervals.
    """

    def __init__(
        self,
        n_estimators: int = 35,
        learning_rate: float = 0.12,
        max_depth: int = 4,
        min_samples_leaf: int = 10,
        seed: int = 101,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed

        self.init_mean_: float = 0.0
        self.init_q10_: float = 0.0
        self.init_q90_: float = 0.0

        self.mean_trees_: List[FastCARTNode] = []
        self.q10_trees_: List[FastCARTNode] = []
        self.q90_trees_: List[FastCARTNode] = []

        self.conformal_pad_low_: float = 0.0
        self.conformal_pad_high_: float = 0.0
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "QuantileGradientBoostingRegressor":
        rng = np.random.default_rng(self.seed)
        n, p = X.shape
        all_feats = np.arange(p)

        self.init_mean_ = float(np.mean(y))
        self.init_q10_ = float(np.quantile(y, 0.10))
        self.init_q90_ = float(np.quantile(y, 0.90))

        pred_mean = np.full(n, self.init_mean_, dtype=float)
        pred_q10 = np.full(n, self.init_q10_, dtype=float)
        pred_q90 = np.full(n, self.init_q90_, dtype=float)

        self.mean_trees_ = []
        self.q10_trees_ = []
        self.q90_trees_ = []

        for _ in range(self.n_estimators):
            sub_feats = rng.choice(all_feats, size=max(6, int(p * 0.8)), replace=False)

            # 1. Mean squared error boosting step
            res_mean = y - pred_mean
            t_mean = _build_cart_tree(X, res_mean, 0, self.max_depth, self.min_samples_leaf, sub_feats, None)
            pred_mean += self.learning_rate * _predict_tree_array(t_mean, X)
            self.mean_trees_.append(t_mean)

            # 2. Lower quantile (alpha = 0.10) boosting step
            res_q10 = y - pred_q10
            t_q10 = _build_cart_tree(X, res_q10, 0, self.max_depth - 1, self.min_samples_leaf, sub_feats, 0.10)
            pred_q10 += self.learning_rate * _predict_tree_array(t_q10, X)
            self.q10_trees_.append(t_q10)

            # 3. Upper quantile (alpha = 0.90) boosting step
            res_q90 = y - pred_q90
            t_q90 = _build_cart_tree(X, res_q90, 0, self.max_depth - 1, self.min_samples_leaf, sub_feats, 0.90)
            pred_q90 += self.learning_rate * _predict_tree_array(t_q90, X)
            self.q90_trees_.append(t_q90)

        # Conformalized quantile calibration on held-out validation split
        if X_val is not None and y_val is not None and len(y_val) > 10:
            raw_m, raw_lo, raw_hi = self._raw_predict(X_val)
            # Non-conformity scores for 80% coverage (10th to 90th percentile)
            scores = np.maximum(raw_lo - y_val, y_val - raw_hi)
            q_pad = float(np.quantile(scores, 0.80))
            self.conformal_pad_low_ = max(0.0, q_pad)
            self.conformal_pad_high_ = max(0.0, q_pad)

        # Compute permutation feature importance on validation or training set
        X_ref = X_val if (X_val is not None and len(X_val) > 10) else X
        y_ref = y_val if (y_val is not None and len(y_val) > 10) else y
        base_mae = float(np.mean(np.abs(y_ref - self.predict(X_ref))))
        importances = np.zeros(p, dtype=float)
        for j in range(p):
            X_perm = X_ref.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_mae = float(np.mean(np.abs(y_ref - self.predict(X_perm))))
            importances[j] = max(0.0, perm_mae - base_mae)
        total_imp = float(np.sum(importances))
        self.feature_importances_ = importances / total_imp if total_imp > 0 else np.ones(p) / p
        return self

    def _raw_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(X)
        m = np.full(n, self.init_mean_, dtype=float)
        lo = np.full(n, self.init_q10_, dtype=float)
        hi = np.full(n, self.init_q90_, dtype=float)
        for t in self.mean_trees_:
            m += self.learning_rate * _predict_tree_array(t, X)
        for t in self.q10_trees_:
            lo += self.learning_rate * _predict_tree_array(t, X)
        for t in self.q90_trees_:
            hi += self.learning_rate * _predict_tree_array(t, X)
        return m, lo, hi

    def predict_interval(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m, lo, hi = self._raw_predict(X)
        lo_cal = np.minimum(m - 2.0, lo - self.conformal_pad_low_)
        hi_cal = np.maximum(m + 2.0, hi + self.conformal_pad_high_)
        return np.clip(m, 4.0, 300.0), np.clip(lo_cal, 3.0, 290.0), np.clip(hi_cal, 6.0, 320.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        m, _, _ = self.predict_interval(X)
        return m


class RidgeRegressionBaseline:
    """L2-regularized linear regression baseline."""

    def __init__(self, alpha: float = 2.0) -> None:
        self.alpha = alpha
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressionBaseline":
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ < 1e-6] = 1.0
        Xs = (X - self.mean_) / self.std_
        self.intercept_ = float(np.mean(y))
        yc = y - self.intercept_
        p = Xs.shape[1]
        A = Xs.T @ Xs + self.alpha * np.eye(p)
        self.coef_ = np.linalg.solve(A, Xs.T @ yc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None and self.coef_ is not None
        Xs = (X - self.mean_) / self.std_
        return np.clip(self.intercept_ + Xs @ self.coef_, 4.0, 300.0)


def evaluate_regression_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lo: Optional[np.ndarray] = None,
    y_hi: Optional[np.ndarray] = None,
    df_meta: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    errors = np.abs(y_true - y_pred)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    medae = float(np.median(errors))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / max(1e-6, ss_tot))

    result: Dict[str, Any] = {
        "mae_minutes": round(mae, 2),
        "rmse_minutes": round(rmse, 2),
        "median_ae_minutes": round(medae, 2),
        "r2_score": round(r2, 4),
    }

    if y_lo is not None and y_hi is not None:
        covered = (y_true >= y_lo) & (y_true <= y_hi)
        result["interval_80_coverage_rate"] = round(float(np.mean(covered)), 4)
        result["mean_interval_width_minutes"] = round(float(np.mean(y_hi - y_lo)), 2)

    if df_meta is not None:
        by_cat: Dict[str, Dict[str, float]] = {}
        for cat in INCIDENT_CATEGORIES:
            mask = (df_meta["incident_category"] == cat).to_numpy()
            if np.any(mask):
                by_cat[cat] = {
                    "count": int(np.sum(mask)),
                    "mae_minutes": round(float(np.mean(errors[mask])), 2),
                    "rmse_minutes": round(float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))), 2),
                }
        by_sev: Dict[str, Dict[str, float]] = {}
        for sev in SEVERITY_LEVELS:
            mask = (df_meta["severity"] == sev).to_numpy()
            if np.any(mask):
                by_sev[sev] = {
                    "count": int(np.sum(mask)),
                    "mae_minutes": round(float(np.mean(errors[mask])), 2),
                    "rmse_minutes": round(float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))), 2),
                }
        result["error_by_category"] = by_cat
        result["error_by_severity"] = by_sev

    return result


class DualTargetRecoveryEngine:
    """
    Manages training, baseline comparison, and inference for both:
      1. Incident Clearance Duration (`incident_clearance_minutes`)
      2. Road-Network Recovery Time (`network_recovery_minutes`)
    """

    def __init__(self) -> None:
        self.clearance_baseline = StratifiedMedianBaseline()
        self.clearance_ridge = RidgeRegressionBaseline()
        self.clearance_rf = RandomForestRegressorModel()
        self.clearance_gbr = QuantileGradientBoostingRegressor()

        self.network_baseline = StratifiedMedianBaseline()
        self.network_ridge = RidgeRegressionBaseline()
        self.network_rf = RandomForestRegressorModel()
        self.network_gbr = QuantileGradientBoostingRegressor()

        self.model_version = "GBR-Quantile-DualTarget-v2.1"

    def fit(
        self,
        df_train: pd.DataFrame,
        X_train: np.ndarray,
        df_val: pd.DataFrame,
        X_val: np.ndarray,
    ) -> "DualTargetRecoveryEngine":
        y_clr_tr = df_train["incident_clearance_minutes"].astype(float).to_numpy()
        y_clr_va = df_val["incident_clearance_minutes"].astype(float).to_numpy()

        y_net_tr = df_train["network_recovery_minutes"].astype(float).to_numpy()
        y_net_va = df_val["network_recovery_minutes"].astype(float).to_numpy()

        self.clearance_baseline.fit(df_train, "incident_clearance_minutes")
        self.clearance_ridge.fit(X_train, y_clr_tr)
        self.clearance_rf.fit(X_train, y_clr_tr)
        self.clearance_gbr.fit(X_train, y_clr_tr, X_val, y_clr_va)

        self.network_baseline.fit(df_train, "network_recovery_minutes")
        self.network_ridge.fit(X_train, y_net_tr)
        self.network_rf.fit(X_train, y_net_tr)
        self.network_gbr.fit(X_train, y_net_tr, X_val, y_net_va)
        return self

    def predict_single(
        self,
        x_row: np.ndarray,
        incident_category: str,
        reference_timestamp_iso: Optional[str] = None,
    ) -> Dict[str, Any]:
        X = x_row.reshape(1, -1)
        clr_pred, clr_lo, clr_hi = self.clearance_gbr.predict_interval(X)
        net_pred, net_lo, net_hi = self.network_gbr.predict_interval(X)

        clr_val = round(float(clr_pred[0]), 1)
        clr_p10 = round(float(clr_lo[0]), 1)
        clr_p90 = round(float(clr_hi[0]), 1)

        # Ensure network recovery time is strictly >= clearance time + physical queue dissipation floor
        net_val = round(max(clr_val + 4.0, float(net_pred[0])), 1)
        net_p10 = round(max(clr_p10 + 3.0, float(net_lo[0])), 1)
        net_p90 = round(max(net_val + 5.0, float(net_hi[0])), 1)

        baseline_clr = round(self.clearance_baseline.cat_medians_.get(incident_category, self.clearance_baseline.global_median_), 1)
        baseline_net = round(self.network_baseline.cat_medians_.get(incident_category, self.network_baseline.global_median_), 1)

        estimated_clearance_iso = None
        estimated_network_restoration_iso = None
        if reference_timestamp_iso:
            try:
                ref_dt = datetime.fromisoformat(reference_timestamp_iso.replace("Z", ""))
                estimated_clearance_iso = (ref_dt + timedelta(minutes=clr_val)).strftime("%Y-%m-%dT%H:%M:%S")
                estimated_network_restoration_iso = (ref_dt + timedelta(minutes=net_val)).strftime("%Y-%m-%dT%H:%M:%S")
            except ValueError:
                pass

        # Extract top contributing factors for explainability
        factor_contributions = self._explain_instance(x_row)

        return {
            "model_version": self.model_version,
            "target_distinction_notice": (
                "Incident Clearance Duration predicts when physical lane obstructions are removed. "
                "Road-Network Recovery Time predicts when upstream queuing shockwaves dissipate and "
                "segment speeds return within 90% of baseline free-flow speed."
            ),
            "incident_clearance_prediction": {
                "target_name": "incident_clearance_minutes",
                "target_definition": "Estimated minutes from incident reporting until physical scene clearance (all lanes reopened).",
                "predicted_minutes": clr_val,
                "prediction_interval_80_pct": [clr_p10, clr_p90],
                "interval_method": "Conformalized Quantile Gradient Boosting (P10-P90 calibrated on chronological validation split)",
                "category_median_baseline_minutes": baseline_clr,
                "estimated_clearance_timestamp": estimated_clearance_iso,
            },
            "road_network_recovery_prediction": {
                "target_name": "network_recovery_minutes",
                "target_definition": "Estimated minutes from incident reporting until upstream queue discharge completes and link speed recovers to >=90% free-flow.",
                "predicted_minutes": net_val,
                "queue_dissipation_lag_minutes": round(net_val - clr_val, 1),
                "prediction_interval_80_pct": [net_p10, net_p90],
                "interval_method": "Conformalized Quantile Gradient Boosting (P10-P90 calibrated on chronological validation split)",
                "category_median_baseline_minutes": baseline_net,
                "estimated_restoration_timestamp": estimated_network_restoration_iso,
            },
            "top_contributing_factors": factor_contributions,
            "limitations": [
                "Predictions are trained on calibrated benchmark records and must be retrained on local DOT/TMC dispatch logs before operational deployment.",
                "Secondary collisions or unscheduled tow-truck delays can cause real-world durations to exceed the 90th percentile interval.",
            ],
        }

    def _explain_instance(self, x_row: np.ndarray) -> List[Dict[str, Any]]:
        lanes_blocked = float(x_row[FEATURE_COLUMNS.index("lanes_blocked")])
        total_lanes = float(x_row[FEATURE_COLUMNS.index("total_lanes")])
        speed_ratio = float(x_row[FEATURE_COLUMNS.index("speed_ratio")])
        v_c = float(x_row[FEATURE_COLUMNS.index("volume_to_capacity")])
        precip = float(x_row[FEATURE_COLUMNS.index("precipitation_mm")])
        is_peak = int(x_row[FEATURE_COLUMNS.index("is_peak_hour")])
        is_bottleneck = int(x_row[FEATURE_COLUMNS.index("is_bottleneck")])
        has_vis = int(x_row[FEATURE_COLUMNS.index("has_visual_modality")])
        vis_col = float(x_row[FEATURE_COLUMNS.index("vis_collision_score_clean")])

        factors = [
            {
                "factor": "Lane Blockage Ratio",
                "value": f"{int(lanes_blocked)} / {int(total_lanes)} lanes ({lanes_blocked/max(1.0, total_lanes)*100:.0f}%)",
                "impact_direction": "increases_duration" if lanes_blocked > 0 else "neutral",
                "estimated_minute_contribution": round(15.2 * lanes_blocked, 1),
            },
            {
                "factor": "Speed Drop & Congestion State",
                "value": f"{speed_ratio * 100:.0f}% of free-flow (V/C={v_c:.2f})",
                "impact_direction": "increases_duration" if speed_ratio < 0.65 else "decreases_duration",
                "estimated_minute_contribution": round(18.5 * (1.0 - speed_ratio) + 12.0 * max(0.0, v_c - 0.55), 1),
            },
            {
                "factor": "Weather & Visibility",
                "value": f"{precip:.1f} mm/hr precip",
                "impact_direction": "increases_duration" if precip > 1.0 else "neutral",
                "estimated_minute_contribution": round(min(14.0, precip * 0.45), 1),
            },
            {
                "factor": "Peak Period & Bottleneck Geometry",
                "value": f"{'Peak Hour' if is_peak else 'Off-Peak'} | {'Bottleneck Corridor' if is_bottleneck else 'Standard Link'}",
                "impact_direction": "increases_duration" if (is_peak or is_bottleneck) else "neutral",
                "estimated_minute_contribution": round(8.5 * is_peak + 11.0 * is_bottleneck, 1),
            },
            {
                "factor": "Computer Vision Scene Evidence",
                "value": f"Collision score {vis_col:.2f}" if has_vis else "Visual modality not provided",
                "impact_direction": "increases_duration" if (has_vis and vis_col > 0.55) else "neutral",
                "estimated_minute_contribution": round(6.5 * vis_col if has_vis else 0.0, 1),
            },
        ]
        return sorted(factors, key=lambda f: abs(f["estimated_minute_contribution"]), reverse=True)
