"""Stacking meta-learner: combines RF + GCN out-of-fold probability
predictions into a final stage prediction. Three interchangeable options
(the meta-learner ablation study in run_sensitivity.py) — default per the
paper is logistic regression."""
from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def build_meta_learner(cfg: Dict[str, Any], meta_type: str | None = None):
    meta_type = meta_type or cfg.get("type", "logistic_regression")
    if meta_type == "logistic_regression":
        p = cfg.get("logistic_regression", {})
        return LogisticRegression(
            C=p.get("C", 1.0), max_iter=p.get("max_iter", 2000), class_weight=p.get("class_weight", "balanced")
        )
    if meta_type == "mlp":
        p = cfg.get("mlp", {})
        return MLPClassifier(
            hidden_layer_sizes=tuple(p.get("hidden_layer_sizes", [32])),
            max_iter=p.get("max_iter", 500),
            alpha=p.get("alpha", 1e-3),
            random_state=42,
        )
    if meta_type == "gradient_boosting":
        p = cfg.get("gradient_boosting", {})
        return GradientBoostingClassifier(
            n_estimators=p.get("n_estimators", 200),
            max_depth=p.get("max_depth", 3),
            learning_rate=p.get("learning_rate", 0.05),
            random_state=42,
        )
    raise ValueError(f"Unknown meta-learner type '{meta_type}'")
