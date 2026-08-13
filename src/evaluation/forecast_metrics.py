"""Impact-forecasting evaluation metrics: ROC-AUC, PR-AUC, F1, and
Capture@k% (of all true positive "reaches IMP within H" instances, what
fraction are contained in the top-k%-highest-scored instances — the
operationally relevant question for an analyst who can only triage a small
percentage of alerts)."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def capture_at_k(y_true: np.ndarray, y_proba: np.ndarray, percent: float) -> float:
    n = len(y_true)
    if n == 0 or y_true.sum() == 0:
        return float("nan")
    k = max(1, int(np.ceil(n * percent / 100.0)))
    top_idx = np.argsort(-y_proba)[:k]
    captured = y_true[top_idx].sum()
    return float(captured / y_true.sum())


def compute_forecast_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, capture_at_percents: List[float], threshold: float = 0.5
) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    out: Dict[str, Any] = {"n_samples": int(len(y_true)), "n_positive": int(y_true.sum())}

    if len(y_true) == 0 or len(set(y_true.tolist())) < 2:
        out.update({"roc_auc": float("nan"), "pr_auc": float("nan"), "f1": float("nan")})
        for p in capture_at_percents:
            out[f"capture_at_{p}pct"] = float("nan")
        out["note"] = "Insufficient class diversity (need both positive and negative instances) to compute ranking metrics."
        return out

    out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    out["pr_auc"] = float(average_precision_score(y_true, y_proba))
    out["f1"] = float(f1_score(y_true, (y_proba >= threshold).astype(int), zero_division=0))
    out["threshold"] = threshold
    for p in capture_at_percents:
        out[f"capture_at_{p}pct"] = capture_at_k(y_true, y_proba, p)
    return out
