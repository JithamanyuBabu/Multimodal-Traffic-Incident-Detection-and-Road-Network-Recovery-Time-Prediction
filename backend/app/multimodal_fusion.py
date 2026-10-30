"""
Multimodal Feature Fusion, Incident Category Classification, and Calibrated Severity Estimation
(Modules D, E, F)
Implements:
  - Class-balanced Multinomial Softmax Classifier solved via L-BFGS-B (`scipy.optimize.minimize`)
  - Temperature-scaled probability calibration
  - Single-Modality Baselines (Structured-Only, Visual-Only)
  - Early Feature-Level Multimodal Fusion (with explicit `has_visual_modality` gating)
  - Confidence-Gated Late Multimodal Fusion
  - Per-class Precision, Recall, F1-score, Confusion Matrix, and Brier Score evaluation
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from .data_pipeline import (
    FEATURE_COLUMNS,
    INCIDENT_CATEGORIES,
    SEVERITY_LEVELS,
    STRUCTURED_ONLY_INDICES,
    VISUAL_ONLY_INDICES,
)


class CalibratedMultinomialClassifier:
    """
    L2-regularized Multinomial Logistic Regression with inverse-frequency class weights
    and post-hoc temperature scaling for calibrated probability estimates.
    """

    def __init__(self, classes: List[str], l2_reg: float = 0.25, class_balanced: bool = True) -> None:
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.l2_reg = l2_reg
        self.class_balanced = class_balanced
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.W_: Optional[np.ndarray] = None
        self.b_: Optional[np.ndarray] = None
        self.temperature_: float = 1.0

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ < 1e-6] = 1.0
        return (X - self.mean_) / self.std_

    def _standardize_transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        return (X - self.mean_) / self.std_

    def fit(self, X: np.ndarray, y_labels: List[str], X_val: Optional[np.ndarray] = None, y_val_labels: Optional[List[str]] = None) -> "CalibratedMultinomialClassifier":
        Xs = self._standardize_fit(X)
        n_samples, n_features = Xs.shape
        n_classes = len(self.classes)
        y_idx = np.array([self.class_to_idx[lbl] for lbl in y_labels], dtype=int)

        # One-hot target matrix
        Y_onehot = np.zeros((n_samples, n_classes), dtype=float)
        Y_onehot[np.arange(n_samples), y_idx] = 1.0

        # Inverse-frequency class weights to handle class imbalance
        if self.class_balanced:
            counts = np.bincount(y_idx, minlength=n_classes).astype(float)
            counts[counts == 0] = 1.0
            class_weights = n_samples / (n_classes * counts)
            sample_weights = class_weights[y_idx]
        else:
            sample_weights = np.ones(n_samples, dtype=float)
        sample_weights = sample_weights / np.mean(sample_weights)

        def loss_and_grad(params: np.ndarray) -> Tuple[float, np.ndarray]:
            W = params[: n_features * n_classes].reshape((n_features, n_classes))
            b = params[n_features * n_classes :]
            logits = Xs @ W + b
            probs = softmax(logits, axis=1)
            log_probs = np.log(np.clip(probs, 1e-12, 1.0))
            ce_loss = -np.mean(sample_weights[:, None] * Y_onehot * log_probs) * n_classes
            reg_loss = 0.5 * self.l2_reg * float(np.sum(W * W)) / n_samples
            diff = (probs - Y_onehot) * sample_weights[:, None] * (n_classes / float(n_samples))
            grad_W = Xs.T @ diff + (self.l2_reg / float(n_samples)) * W
            grad_b = np.sum(diff, axis=0)
            return float(ce_loss + reg_loss), np.concatenate([grad_W.ravel(), grad_b])

        init_params = np.zeros(n_features * n_classes + n_classes, dtype=float)
        res = minimize(loss_and_grad, init_params, method="L-BFGS-B", jac=True, options={"maxiter": 250})
        self.W_ = res.x[: n_features * n_classes].reshape((n_features, n_classes))
        self.b_ = res.x[n_features * n_classes :]

        # Temperature calibration on validation set if provided
        if X_val is not None and y_val_labels is not None and len(y_val_labels) > 5:
            Xv = self._standardize_transform(X_val)
            val_logits = Xv @ self.W_ + self.b_
            yv_idx = np.array([self.class_to_idx[lbl] for lbl in y_val_labels], dtype=int)

            def nll_temp(t_arr: np.ndarray) -> float:
                t = max(0.15, float(t_arr[0]))
                p = softmax(val_logits / t, axis=1)
                return float(-np.mean(np.log(np.clip(p[np.arange(len(yv_idx)), yv_idx], 1e-12, 1.0))))

            t_res = minimize(nll_temp, np.array([1.0]), bounds=[(0.2, 5.0)], method="L-BFGS-B")
            self.temperature_ = round(float(t_res.x[0]), 4)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.W_ is not None and self.b_ is not None
        Xs = self._standardize_transform(X)
        logits = (Xs @ self.W_ + self.b_) / max(0.15, self.temperature_)
        return softmax(logits, axis=1)

    def predict(self, X: np.ndarray) -> List[str]:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return [self.classes[int(i)] for i in idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classes": self.classes,
            "mean": self.mean_.tolist() if self.mean_ is not None else [],
            "std": self.std_.tolist() if self.std_ is not None else [],
            "W": self.W_.tolist() if self.W_ is not None else [],
            "b": self.b_.tolist() if self.b_ is not None else [],
            "temperature": self.temperature_,
        }


class MultimodalFusionPipeline:
    """
    Implements Early Feature-Level Fusion and Confidence-Gated Late Fusion across
    Structured Traffic/Weather/Road features and Computer Vision scene embeddings.
    Explicitly handles missing visual modalities without silent imputation.
    """

    def __init__(self) -> None:
        self.cat_structured = CalibratedMultinomialClassifier(INCIDENT_CATEGORIES, l2_reg=0.3)
        self.cat_visual = CalibratedMultinomialClassifier(INCIDENT_CATEGORIES, l2_reg=0.3)
        self.cat_early_fusion = CalibratedMultinomialClassifier(INCIDENT_CATEGORIES, l2_reg=0.25)

        self.sev_structured = CalibratedMultinomialClassifier(SEVERITY_LEVELS, l2_reg=0.3)
        self.sev_visual = CalibratedMultinomialClassifier(SEVERITY_LEVELS, l2_reg=0.3)
        self.sev_early_fusion = CalibratedMultinomialClassifier(SEVERITY_LEVELS, l2_reg=0.25)

        self.late_fusion_alpha_cat: float = 0.62  # Weight on structured vs visual when visual is present
        self.late_fusion_alpha_sev: float = 0.68

    def fit(
        self,
        X_train: np.ndarray,
        y_cat_train: List[str],
        y_sev_train: List[str],
        X_val: np.ndarray,
        y_cat_val: List[str],
        y_sev_val: List[str],
    ) -> "MultimodalFusionPipeline":
        X_tr_struct = X_train[:, STRUCTURED_ONLY_INDICES]
        X_va_struct = X_val[:, STRUCTURED_ONLY_INDICES]

        X_tr_vis = X_train[:, VISUAL_ONLY_INDICES]
        X_va_vis = X_val[:, VISUAL_ONLY_INDICES]

        self.cat_structured.fit(X_tr_struct, y_cat_train, X_va_struct, y_cat_val)
        self.cat_visual.fit(X_tr_vis, y_cat_train, X_va_vis, y_cat_val)
        self.cat_early_fusion.fit(X_train, y_cat_train, X_val, y_cat_val)

        self.sev_structured.fit(X_tr_struct, y_sev_train, X_va_struct, y_sev_val)
        self.sev_visual.fit(X_tr_vis, y_sev_train, X_va_vis, y_sev_val)
        self.sev_early_fusion.fit(X_train, y_sev_train, X_val, y_sev_val)
        return self

    def predict_late_fusion_proba(self, X: np.ndarray, task: str = "category") -> np.ndarray:
        """
        Confidence-Gated Late Fusion:
        When `has_visual_modality == 0`, relies 100% on the Structured-Only model
        and flags modality absence explicitly. When `has_visual_modality == 1`, gates
        the visual model contribution by `vis_confidence`.
        """
        has_vis_col = FEATURE_COLUMNS.index("has_visual_modality")
        vis_conf_col = FEATURE_COLUMNS.index("vis_confidence")
        has_vis = X[:, has_vis_col]
        vis_conf = np.clip(X[:, vis_conf_col], 0.0, 1.0)

        X_struct = X[:, STRUCTURED_ONLY_INDICES]
        X_vis = X[:, VISUAL_ONLY_INDICES]

        if task == "category":
            p_struct = self.cat_structured.predict_proba(X_struct)
            p_vis = self.cat_visual.predict_proba(X_vis)
            p_early = self.cat_early_fusion.predict_proba(X)
            base_vis_weight = 0.38
        else:
            p_struct = self.sev_structured.predict_proba(X_struct)
            p_vis = self.sev_visual.predict_proba(X_vis)
            p_early = self.sev_early_fusion.predict_proba(X)
            base_vis_weight = 0.28

        # Effective visual gate is 0 when visual modality is absent
        w_vis = (has_vis > 0).astype(float) * base_vis_weight * (0.5 + 0.5 * vis_conf)
        w_early = (has_vis > 0).astype(float) * 0.45
        w_struct = 1.0 - w_vis - w_early

        combined = (
            w_struct[:, None] * p_struct
            + w_vis[:, None] * p_vis
            + w_early[:, None] * p_early
        )
        return combined / np.sum(combined, axis=1, keepdims=True)

    def predict_single(self, x_row: np.ndarray) -> Dict[str, Any]:
        X = x_row.reshape(1, -1)
        has_vis = bool(X[0, FEATURE_COLUMNS.index("has_visual_modality")] > 0.5)

        cat_probs = self.predict_late_fusion_proba(X, task="category")[0]
        sev_probs = self.predict_late_fusion_proba(X, task="severity")[0]

        cat_idx = int(np.argmax(cat_probs))
        sev_idx = int(np.argmax(sev_probs))

        # Compute Shannon entropy as calibrated uncertainty index (0 = certain, 1 = max uniform uncertainty)
        cat_entropy = float(-np.sum(cat_probs * np.log(np.clip(cat_probs, 1e-12, 1.0))) / np.log(len(INCIDENT_CATEGORIES)))
        sev_entropy = float(-np.sum(sev_probs * np.log(np.clip(sev_probs, 1e-12, 1.0))) / np.log(len(SEVERITY_LEVELS)))

        return {
            "fusion_method": "confidence_gated_hybrid_multimodal" if has_vis else "structured_fallback_missing_visual",
            "visual_modality_present": has_vis,
            "missing_modalities": [] if has_vis else ["computer_vision_scene_features"],
            "incident_category": {
                "predicted_class": INCIDENT_CATEGORIES[cat_idx],
                "confidence": round(float(cat_probs[cat_idx]), 3),
                "normalized_entropy_uncertainty": round(cat_entropy, 3),
                "probabilities": {c: round(float(p), 4) for c, p in zip(INCIDENT_CATEGORIES, cat_probs)},
            },
            "severity": {
                "predicted_severity": SEVERITY_LEVELS[sev_idx],
                "confidence": round(float(sev_probs[sev_idx]), 3),
                "normalized_entropy_uncertainty": round(sev_entropy, 3),
                "probabilities": {s: round(float(p), 4) for s, p in zip(SEVERITY_LEVELS, sev_probs)},
                "limitation_notice": (
                    "Severity represents estimated operational traffic disruption level (low/moderate/high/critical) "
                    "derived from lane blockage, speed drop, and scene cues. It is not a clinical or emergency triage assessment."
                ),
            },
        }


def compute_classification_metrics(
    y_true: List[str], y_pred_probs: np.ndarray, classes: List[str]
) -> Dict[str, Any]:
    """
    Computes per-class Precision, Recall, F1-score, Support, Macro-F1, Weighted-F1,
    Accuracy, Confusion Matrix, and Multi-class Brier Calibration Score.
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_true_idx = np.array([class_to_idx[c] for c in y_true], dtype=int)
    y_pred_idx = np.argmax(y_pred_probs, axis=1)
    n = len(y_true_idx)
    k = len(classes)

    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true_idx, y_pred_idx):
        cm[t, p] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    f1s: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    supports: List[int] = []

    for i, c in enumerate(classes):
        tp = float(cm[i, i])
        fp = float(np.sum(cm[:, i]) - tp)
        fn = float(np.sum(cm[i, :]) - tp)
        sup = int(np.sum(cm[i, :]))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": sup,
        }
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(sup)

    accuracy = float(np.trace(cm)) / max(1, n)
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(np.sum(np.array(f1s) * np.array(supports)) / max(1, np.sum(supports)))

    # Multi-class Brier score
    Y_oh = np.zeros((n, k), dtype=float)
    Y_oh[np.arange(n), y_true_idx] = 1.0
    brier = float(np.mean(np.sum((y_pred_probs - Y_oh) ** 2, axis=1)))

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(float(np.mean(precisions)), 4),
        "macro_recall": round(float(np.mean(recalls)), 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "brier_calibration_score": round(brier, 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": classes,
            "matrix": cm.tolist(),
        },
    }
