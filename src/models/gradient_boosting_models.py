"""Gradient-boosting tabular baselines (XGBoost / LightGBM) — independent
feature-only baselines in the model-comparison table, trained on the same
X (window/record aggregated features) as the Random Forest baseline."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def build_xgboost(cfg: Dict[str, Any], num_classes: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=cfg.get("n_estimators", 300),
        max_depth=cfg.get("max_depth", 6),
        learning_rate=cfg.get("learning_rate", 0.1),
        subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample_bytree", 0.8),
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=cfg.get("random_state", 42),
        n_jobs=cfg.get("n_jobs", -1),
    )


def build_lightgbm(cfg: Dict[str, Any], num_classes: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=cfg.get("n_estimators", 300),
        max_depth=cfg.get("max_depth", -1),
        learning_rate=cfg.get("learning_rate", 0.1),
        subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample_bytree", 0.8),
        objective="multiclass",
        num_class=num_classes,
        class_weight="balanced",
        random_state=cfg.get("random_state", 42),
        n_jobs=cfg.get("n_jobs", -1),
        verbosity=-1,
    )


def fit_predict_proba_aligned(model, X_train, y_train, X_eval, num_classes: int) -> np.ndarray:
    """Fit `model` (sklearn-API classifier) and return predict_proba on
    X_eval aligned to `num_classes` columns (pads any class absent from the
    training fold with a zero column), matching
    src/training/stage_detector_trainer.py::_align_proba's contract."""
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_eval)
    classes_seen = np.asarray(model.classes_)
    if len(classes_seen) == num_classes:
        return proba
    out = np.zeros((proba.shape[0], num_classes), dtype=proba.dtype)
    out[:, classes_seen.astype(int)] = proba
    return out
