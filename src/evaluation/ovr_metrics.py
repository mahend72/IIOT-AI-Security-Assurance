"""One-vs-rest stage-alerting metrics (IAD-vs-rest, LMEP-vs-rest, IMP-vs-rest).

Threshold-selection rule (documented, not hardcoded silently): for each
stage, the alerting threshold is the value in [0.01, 0.99] (step 0.01) that
maximizes F1 for that stage's one-vs-rest problem ON THE VALIDATION SPLIT
ONLY. That threshold is then applied, unchanged, to the TEST split to
report precision/recall/F1 -- test data is never used to select a
threshold, per the project's no-test-leakage rule."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

THRESHOLD_SELECTION_RULE = (
    "threshold in [0.01, 0.99] (step 0.01) maximizing one-vs-rest F1 on the VALIDATION split; "
    "applied unchanged to TEST (test never used for threshold selection)"
)


def _select_threshold_on_val(y_val: np.ndarray, p_val: np.ndarray) -> float:
    if len(y_val) == 0 or y_val.sum() == 0:
        return 0.5
    grid = np.arange(0.01, 1.0, 0.01)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_val, (p_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def one_vs_rest_alerting_metrics(
    y_true: Dict[str, np.ndarray],
    proba: Dict[str, np.ndarray],
    class_names: List[str],
    stages: List[str],
) -> List[Dict[str, Any]]:
    """y_true / proba: {split: array}, proba is [N, num_classes] aligned to
    `class_names`. Returns one row per (stage, split)."""
    rows = []
    for stage in stages:
        cls_idx = class_names.index(stage)
        y_val_bin = (y_true["val"] == cls_idx).astype(int) if len(y_true.get("val", [])) else np.array([])
        p_val = proba["val"][:, cls_idx] if len(y_val_bin) else np.array([])
        threshold = _select_threshold_on_val(y_val_bin, p_val)

        for split in ("train", "val", "test"):
            yt = y_true.get(split, np.array([]))
            pr = proba.get(split, np.zeros((0, len(class_names))))
            if len(yt) == 0:
                rows.append({"stage": stage, "split": split, "precision": float("nan"), "recall": float("nan"),
                             "f1": float("nan"), "pr_auc": float("nan"), "threshold": threshold,
                             "threshold_selection_rule": THRESHOLD_SELECTION_RULE, "n_positive": 0, "n_total": 0})
                continue
            y_bin = (yt == cls_idx).astype(int)
            p = pr[:, cls_idx]
            y_pred = (p >= threshold).astype(int)
            n_pos = int(y_bin.sum())
            pr_auc = float(average_precision_score(y_bin, p)) if n_pos > 0 and n_pos < len(y_bin) else float("nan")
            rows.append({
                "stage": stage, "split": split,
                "precision": float(precision_score(y_bin, y_pred, zero_division=0)),
                "recall": float(recall_score(y_bin, y_pred, zero_division=0)),
                "f1": float(f1_score(y_bin, y_pred, zero_division=0)),
                "pr_auc": pr_auc,
                "threshold": threshold,
                "threshold_selection_rule": THRESHOLD_SELECTION_RULE,
                "n_positive": n_pos, "n_total": int(len(y_bin)),
            })
    return rows
