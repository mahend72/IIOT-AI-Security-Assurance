"""Stage-detection evaluation metrics: precision/recall/F1 per class,
macro-F1, and confusion matrix."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def compute_stage_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_f1 = float(np.mean(f1))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    }
    return {
        "per_class": per_class,
        "macro_f1": macro_f1,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "n_samples": int(len(y_true)),
    }
