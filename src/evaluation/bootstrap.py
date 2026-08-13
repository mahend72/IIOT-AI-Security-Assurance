"""Bootstrap confidence intervals for any metric function, computed by
resampling instances (with replacement) and recomputing the metric.
Used for both stage-detection (macro-F1) and impact-forecasting
(ROC-AUC / PR-AUC / F1 / Capture@k) metrics wherever the project spec asks
for CIs.

Note: bootstrap resamples INSTANCES (already-computed predictions), not
assets and not raw data — it does not re-fit any model, re-scale any
feature, or otherwise touch train data, so it cannot itself introduce
leakage. It quantifies sampling variability of the evaluation set only.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_iterations: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    if n == 0:
        return {"point": float("nan"), "lower": float("nan"), "upper": float("nan"), "n_valid_iterations": 0}

    rng = np.random.default_rng(seed)
    point = float(metric_fn(y_true, y_score))

    samples = []
    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        try:
            val = metric_fn(yt, ys)
            if np.isfinite(val):
                samples.append(val)
        except (ValueError, ZeroDivisionError):
            continue  # e.g. a resample with only one class present for an AUC metric

    alpha = (1 - ci) / 2
    if samples:
        lower = float(np.quantile(samples, alpha))
        upper = float(np.quantile(samples, 1 - alpha))
    else:
        lower, upper = float("nan"), float("nan")

    return {"point": point, "lower": lower, "upper": upper, "ci": ci, "n_valid_iterations": len(samples)}


def asset_bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_score: np.ndarray,
    asset_ids: np.ndarray,
    n_iterations: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Same contract as `bootstrap_ci`, but resamples whole ASSETS with
    replacement (not individual instances) -- every instance belonging to a
    sampled asset is included together each draw. Used wherever an asset
    can contribute more than one instance to a split, so instance-level
    resampling would understate variance by treating correlated same-asset
    instances as independent draws."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    asset_ids = np.asarray(asset_ids)
    n = len(y_true)
    if n == 0:
        return {"point": float("nan"), "lower": float("nan"), "upper": float("nan"), "n_valid_iterations": 0}

    unique_assets = np.unique(asset_ids)
    asset_to_rows = {a: np.nonzero(asset_ids == a)[0] for a in unique_assets}
    rng = np.random.default_rng(seed)
    point = float(metric_fn(y_true, y_score))

    samples = []
    for _ in range(n_iterations):
        sampled_assets = unique_assets[rng.integers(0, len(unique_assets), size=len(unique_assets))]
        idx = np.concatenate([asset_to_rows[a] for a in sampled_assets])
        yt, ys = y_true[idx], y_score[idx]
        try:
            val = metric_fn(yt, ys)
            if np.isfinite(val):
                samples.append(val)
        except (ValueError, ZeroDivisionError):
            continue

    alpha = (1 - ci) / 2
    if samples:
        lower = float(np.quantile(samples, alpha))
        upper = float(np.quantile(samples, 1 - alpha))
    else:
        lower, upper = float("nan"), float("nan")

    return {
        "point": point, "lower": lower, "upper": upper, "ci": ci,
        "n_valid_iterations": len(samples), "n_unique_assets": len(unique_assets),
    }
