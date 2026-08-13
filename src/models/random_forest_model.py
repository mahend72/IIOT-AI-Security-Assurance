"""Random Forest tabular stage-detector — one of the two base learners feeding
the stacking meta-learner (src/training/stage_detector_trainer.py)."""
from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier


def build_random_forest(cfg: Dict[str, Any]) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.get("n_estimators", 300),
        max_depth=cfg.get("max_depth", None),
        min_samples_leaf=cfg.get("min_samples_leaf", 2),
        class_weight=cfg.get("class_weight", "balanced"),
        random_state=cfg.get("random_state", 42),
        n_jobs=cfg.get("n_jobs", -1),
    )
