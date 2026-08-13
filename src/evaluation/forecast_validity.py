"""Second, INSTANCE-LEVEL impact-forecasting validity gate.

The dataset adapters' `impact_forecasting_valid` (src/data/*.py) checks
upfront metadata (asset cardinality, timestamp parseability/calendar-date
fraction). That is necessary but NOT sufficient: even with a perfectly
trustworthy timestamp and plenty of distinct assets, `build_forecast_instances`
(src/training/sequence_builder.py) can still structurally produce zero (or
too few / single-class-only) usable pre-impact sequences -- which is exactly
what happens on real ToN-IoT: at every candidate window size (30-300s),
100% of assets are observed within a single Delta-t-window (their
attack/device sessions are shorter than even the smallest candidate
window), so there is no earlier "cut point" strictly before an asset's
first IMP-labeled window and no later window to confirm a negative -- zero
forecasting instances are constructed, at every window size. This gate
catches that (and any other degenerate case: zero instances in train, or a
single-class-only split that would make ROC-AUC/PR-AUC undefined) BEFORE
training, so callers write a SKIPPED_WITH_REASON report instead of hitting
the trainer's hard ValueError or silently emitting NaN-filled metrics.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from src.training.sequence_builder import ForecastInstance


def check_instance_level_validity(instances: List[ForecastInstance], min_instances: int = 10) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if len(instances) == 0:
        reasons.append(
            "zero pre-impact forecasting instances could be constructed at any split -- every asset's "
            "observed activity fits inside a single Delta-t window (no earlier 'cut point' exists before "
            "the asset's first IMP window, and/or no later window exists to confirm a negative)."
        )
        return False, reasons

    labels = np.array([i.label for i in instances])
    splits = np.array([i.split for i in instances])

    n_train = int((splits == "train").sum())
    if n_train < min_instances:
        reasons.append(f"only {n_train} TRAIN instances (< required {min_instances}) -- insufficient to fit a forecaster.")

    for split in ("train", "test"):
        mask = splits == split
        n = int(mask.sum())
        if n == 0:
            reasons.append(f"zero {split.upper()} instances.")
            continue
        n_pos = int(labels[mask].sum())
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            reasons.append(
                f"{split.upper()} split has only one class present ({n_pos} positive / {n_neg} negative of {n} "
                f"instances) -- ROC-AUC/PR-AUC/Capture@k are undefined without both classes."
            )

    return len(reasons) == 0, reasons


def instance_summary(instances: List[ForecastInstance]) -> Dict[str, Any]:
    labels = np.array([i.label for i in instances]) if instances else np.array([])
    splits = np.array([i.split for i in instances]) if instances else np.array([])
    out = {"n_total": len(instances), "n_positive_total": int(labels.sum()) if len(labels) else 0}
    for split in ("train", "val", "test"):
        mask = splits == split
        out[f"n_{split}"] = int(mask.sum())
        out[f"n_positive_{split}"] = int(labels[mask].sum()) if mask.sum() else 0
    return out
