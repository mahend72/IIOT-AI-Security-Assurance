"""Trains every model required by the manuscript's stage-detection
comparison table (RF, GCN, late fusion, stacked RF-GCN, XGBoost, LightGBM,
GRU/LSTM no-graph temporal, GraphSAGE, GAT) across one or more seeds, and
returns per-model per-seed per-split (y_true, proba) so the reporting
script can compute macro-F1 / per-class F1 mean+-std and one-vs-rest
alerting uniformly for every model.

"Late fusion" = simple average of the RF and GCN probabilities from the
SAME stacking run (no separate training) -- contrasts with "Stacked_RF_GCN"
(a trained logistic-regression meta-learner over out-of-fold RF+GCN
predictions, src/training/stage_detector_trainer.py).

GraphSAGE / GAT are trained as standalone final-fit-only graph models (same
protocol as the GCN "final model" branch of `train_stage_detector`), NOT
via the expensive out-of-fold stacking cross-validation -- that machinery
exists to produce leakage-free meta-features for the stacking meta-learner
specifically, which these standalone comparison-table rows do not need.

The no-graph temporal GRU/LSTM baseline is skipped (not silently run on
untrustworthy row order) whenever `prepared.used_asset_level_fallback` is
True -- see src/training/temporal_baseline_trainer.py's docstring.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.mapping.label_mapper import STAGE_ORDER
from src.models.gradient_boosting_models import build_lightgbm, build_xgboost
from src.training.gcn_trainer import gcn_predict_proba, train_gcn
from src.training.stage_detector_trainer import train_stage_detector
from src.training.temporal_baseline_trainer import train_temporal_baseline
from src.utils.logging_utils import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

NUM_CLASSES = len(STAGE_ORDER)

# Models that require a trustworthy per-asset window ORDER to be meaningful
# (contrast with e.g. RF/XGBoost/LightGBM, which only need the window's own
# aggregated features, order-independent).
ORDER_DEPENDENT_MODELS = {"GRU_Temporal_NoGraph"}


def _align_proba(proba: np.ndarray, classes_seen: np.ndarray, num_classes: int) -> np.ndarray:
    if len(classes_seen) == num_classes:
        return proba
    out = np.zeros((proba.shape[0], num_classes), dtype=proba.dtype)
    out[:, classes_seen.astype(int)] = proba
    return out


def _standalone_gnn(prepared, conv_type: str, seed: int) -> Dict[str, Dict[str, np.ndarray]]:
    data_full = prepared.graph.to_pyg_data(prepared.X, prepared.y, mode="both")
    model, _ = train_gcn(
        data_full, data_full.train_mask, data_full.val_mask, NUM_CLASSES,
        prepared.cfg["models"]["gcn"], seed=seed, conv_type=conv_type,
    )
    proba_all = gcn_predict_proba(model, data_full)
    split = prepared.windows_df["split"].to_numpy()
    return {
        name: {"y_true": prepared.y[split == name], "proba": proba_all[split == name]}
        for name in ("train", "val", "test")
    }


def run_all_stage_models(
    prepared,
    seeds: List[int],
    n_folds: int = 5,
    include_temporal_baseline: bool = True,
    models: List[str] | None = None,
) -> Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]]:
    """Returns {model_name: {seed: {split: {'y_true':..., 'proba':...}}}}.

    `models`: optional whitelist (e.g. to skip GraphSAGE/GAT in a quick
    sensitivity sweep that only needs RF/GCN/Stacked) -- default None runs
    everything.
    """
    results: Dict[str, Dict[int, Dict[str, Any]]] = {}
    want = (lambda name: models is None or name in models)

    for seed in seeds:
        set_global_seed(seed)
        logger.info(f"[model_comparison] === seed {seed} ===")

        need_stacking_run = any(want(n) for n in ("RF", "GCN", "Stacked_RF_GCN", "Late_Fusion_RF_GCN"))
        if need_stacking_run:
            sd = train_stage_detector(
                prepared.windows_df, prepared.X, prepared.y, prepared.graph,
                prepared.cfg, n_folds=n_folds, seed=seed,
            )
            for model_name, proba_dict in [("RF", sd.rf_proba), ("GCN", sd.gcn_proba), ("Stacked_RF_GCN", sd.stacked_proba)]:
                if want(model_name):
                    results.setdefault(model_name, {})[seed] = {
                        split: {"y_true": sd.y_true[split], "proba": proba_dict[split]} for split in ("train", "val", "test")
                    }
            if want("Late_Fusion_RF_GCN"):
                results.setdefault("Late_Fusion_RF_GCN", {})[seed] = {
                    split: {"y_true": sd.y_true[split], "proba": (sd.rf_proba[split] + sd.gcn_proba[split]) / 2.0}
                    for split in ("train", "val", "test")
                }

        split = prepared.windows_df["split"].to_numpy()
        train_mask, val_mask, test_mask = split == "train", split == "val", split == "test"

        for name, builder_key in [("XGBoost", "xgboost"), ("LightGBM", "lightgbm")]:
            if not want(name):
                continue
            builder = build_xgboost if builder_key == "xgboost" else build_lightgbm
            model_cfg = {**prepared.cfg["models"][builder_key], "random_state": seed}
            model = builder(model_cfg, NUM_CLASSES)
            model.fit(prepared.X[train_mask], prepared.y[train_mask])
            out = {}
            for sname, smask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
                proba = model.predict_proba(prepared.X[smask])
                proba = _align_proba(proba, np.asarray(model.classes_), NUM_CLASSES)
                out[sname] = {"y_true": prepared.y[smask], "proba": proba}
            results.setdefault(name, {})[seed] = out

        for name, conv in [("GraphSAGE", "sage"), ("GAT", "gat")]:
            if want(name):
                results.setdefault(name, {})[seed] = _standalone_gnn(prepared, conv, seed)

        if want("GRU_Temporal_NoGraph"):
            if include_temporal_baseline and not prepared.used_asset_level_fallback:
                tb = train_temporal_baseline(prepared.windows_df, prepared.X, prepared.y, NUM_CLASSES, prepared.cfg, seed=seed)
                results.setdefault("GRU_Temporal_NoGraph", {})[seed] = {
                    split: {"y_true": tb.y_true[split], "proba": tb.proba[split]} for split in ("train", "val", "test")
                }
            else:
                logger.info(
                    f"[model_comparison] Skipping GRU_Temporal_NoGraph for seed {seed}: "
                    f"dataset uses the record-level fallback (no trustworthy window order)."
                )

    return results
