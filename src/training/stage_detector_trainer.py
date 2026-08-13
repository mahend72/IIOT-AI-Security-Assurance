"""Stage detector: Random Forest + GCN base learners, combined by a
logistic-regression (or MLP / gradient-boosting) stacking meta-learner
trained on OUT-OF-FOLD, ASSET-DISJOINT base-learner predictions.

Why out-of-fold: if the meta-learner were trained on RF's/GCN's predictions
on the very data those models were fit on, it would just learn to trust
whichever base learner overfits hardest — not which one actually
generalizes. Standard stacked generalization instead cross-validates the
base learners over the TRAIN split, using each fold's held-out predictions
as unbiased-ish meta-features, then fits the final base learners on the
whole train split for use at val/test time.

Why asset-disjoint folds specifically: exactly the same reason the overall
train/val/test split is asset-disjoint (src/preprocessing/splitting.py) —
two windows of the same asset are correlated, so a random (record-level)
fold split would leak.

GCN-specific note: during out-of-fold generation, each fold's GCN is
trained on a subgraph containing ONLY train-split nodes (built via
AssetTimeGraph.induced_subgraph) — val/test node features are structurally
unreachable during this phase, not just their labels. The FINAL GCN (used
to score val/test) is trained with gradients only from train nodes but its
forward pass runs over the full graph, which is the standard transductive
GNN setting: val/test features are visible at inference time (as they
would be in production), but never influence training loss or any
preprocessing/threshold decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from src.data.schema import ASSET_ID_COL
from src.graph.graph_builder import AssetTimeGraph, GraphMode, STAGE_TO_INT
from src.mapping.label_mapper import STAGE_ORDER
from src.models.meta_learner import build_meta_learner
from src.models.random_forest_model import build_random_forest
from src.training.gcn_trainer import gcn_predict_proba, train_gcn
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

NUM_CLASSES = len(STAGE_ORDER)


def _align_proba(proba: np.ndarray, classes_seen: np.ndarray, num_classes: int) -> np.ndarray:
    """RandomForest.predict_proba only returns columns for classes present
    in that fit's training data. On small folds a rare class (e.g. IMP) can
    be entirely absent — pad with zeros in the right column so every fold's
    output aligns to the same [n, num_classes] layout."""
    if len(classes_seen) == num_classes:
        return proba
    out = np.zeros((proba.shape[0], num_classes), dtype=proba.dtype)
    out[:, classes_seen.astype(int)] = proba
    return out


@dataclass
class StageDetectorResult:
    class_names: list = field(default_factory=lambda: list(STAGE_ORDER))
    rf_model: Any = None
    gcn_model: Any = None
    meta_learner: Any = None
    y_true: Dict[str, np.ndarray] = field(default_factory=dict)  # split -> labels
    rf_proba: Dict[str, np.ndarray] = field(default_factory=dict)
    gcn_proba: Dict[str, np.ndarray] = field(default_factory=dict)
    stacked_proba: Dict[str, np.ndarray] = field(default_factory=dict)
    gcn_history: Dict[str, list] = field(default_factory=dict)
    oof_meta_features: Optional[np.ndarray] = None
    oof_labels: Optional[np.ndarray] = None


def train_stage_detector(
    windows_df,
    X: np.ndarray,
    y: np.ndarray,
    graph: AssetTimeGraph,
    cfg: Dict[str, Any],
    graph_mode: GraphMode = "both",
    n_folds: int = 5,
    seed: int = 42,
    meta_type: Optional[str] = None,
) -> StageDetectorResult:
    result = StageDetectorResult()
    split = windows_df["split"].to_numpy()
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    # `seed` overrides rf_cfg's static random_state so a multi-seed sweep
    # (src/training/model_comparison.py) actually varies RF's bagging/
    # feature-subsampling randomness -- otherwise every "seed" would fit a
    # bit-for-bit identical RF (std=0 across seeds is then a config
    # artifact, not evidence of stability).
    rf_cfg = {**cfg["models"]["random_forest"], "random_state": seed}
    gcn_cfg = cfg["models"]["gcn"]
    meta_cfg = cfg["models"]["meta_learner"]

    # ---- 1. Final base learners, fit on the WHOLE train split -----------------
    logger.info("Fitting final Random Forest on full train split...")
    rf_full = build_random_forest(rf_cfg)
    rf_full.fit(X[train_mask], y[train_mask])
    result.rf_model = rf_full

    logger.info("Fitting final GCN on full train split (transductive forward pass over full graph)...")
    data_full = graph.to_pyg_data(X, y, mode=graph_mode)
    gcn_full, gcn_history = train_gcn(
        data_full, data_full.train_mask, data_full.val_mask, NUM_CLASSES, gcn_cfg, seed=seed, verbose=False
    )
    result.gcn_model = gcn_full
    result.gcn_history = gcn_history
    gcn_proba_all = gcn_predict_proba(gcn_full, data_full)

    for name, mask in [("val", val_mask), ("test", test_mask)]:
        result.y_true[name] = y[mask]
        result.rf_proba[name] = _align_proba(rf_full.predict_proba(X[mask]), rf_full.classes_, NUM_CLASSES)
        result.gcn_proba[name] = gcn_proba_all[mask]

    # ---- 2. Out-of-fold base-learner predictions on TRAIN, asset-disjoint -----
    train_indices = np.nonzero(train_mask)[0]
    asset_ids_train = windows_df.iloc[train_indices][ASSET_ID_COL].to_numpy()
    n_folds_eff = min(n_folds, len(set(asset_ids_train)))
    if n_folds_eff < 2:
        raise ValueError("Need at least 2 distinct train assets to build out-of-fold stacking features.")
    gkf = GroupKFold(n_splits=n_folds_eff)

    rf_oof = np.zeros((len(train_indices), NUM_CLASSES))
    gcn_oof = np.zeros((len(train_indices), NUM_CLASSES))

    # Train-only induced subgraph: val/test node FEATURES are structurally
    # unreachable from here, not just their labels (see module docstring).
    sub_data, sub_global_idx = graph.induced_subgraph(X, y, keep_row_mask=train_mask, mode=graph_mode)
    assert np.array_equal(sub_global_idx, train_indices), "induced_subgraph ordering must match train_indices"

    logger.info(f"Generating out-of-fold stacking features with {n_folds_eff}-fold asset-disjoint CV...")
    for fold_i, (fold_train_pos, fold_heldout_pos) in enumerate(gkf.split(train_indices, groups=asset_ids_train)):
        fold_train_global = train_indices[fold_train_pos]
        fold_heldout_global = train_indices[fold_heldout_pos]

        # -- RF fold --
        rf_fold = build_random_forest(rf_cfg)
        rf_fold.fit(X[fold_train_global], y[fold_train_global])
        proba = rf_fold.predict_proba(X[fold_heldout_global])
        rf_oof[fold_heldout_pos] = _align_proba(proba, rf_fold.classes_, NUM_CLASSES)

        # -- GCN fold (trained on the train-only subgraph; forward pass at
        # inference reads out ALL train-only nodes, we just keep this fold's) --
        train_mask_local = torch.zeros(len(sub_global_idx), dtype=torch.bool)
        train_mask_local[fold_train_pos] = True
        stop_mask_local = torch.zeros(len(sub_global_idx), dtype=torch.bool)
        stop_mask_local[fold_heldout_pos] = True
        gcn_fold, _ = train_gcn(sub_data, train_mask_local, stop_mask_local, NUM_CLASSES, gcn_cfg, seed=seed + fold_i + 1)
        fold_proba_all = gcn_predict_proba(gcn_fold, sub_data)
        gcn_oof[fold_heldout_pos] = fold_proba_all[fold_heldout_pos]

        logger.info(f"  fold {fold_i + 1}/{n_folds_eff}: {len(fold_heldout_global)} held-out train instances scored")

    result.oof_meta_features = np.concatenate([rf_oof, gcn_oof], axis=1)
    result.oof_labels = y[train_indices]
    result.y_true["train"] = result.oof_labels
    result.rf_proba["train"] = rf_oof
    result.gcn_proba["train"] = gcn_oof

    # ---- 3. Fit the stacking meta-learner on OOF train features ---------------
    logger.info(f"Fitting stacking meta-learner (type={meta_type or meta_cfg.get('type')})...")
    meta_learner = build_meta_learner(meta_cfg, meta_type=meta_type)
    meta_learner.fit(result.oof_meta_features, result.oof_labels)
    result.meta_learner = meta_learner

    meta_classes = np.asarray(meta_learner.classes_)
    for name in ("train", "val", "test"):
        meta_features = np.concatenate([result.rf_proba[name], result.gcn_proba[name]], axis=1)
        raw_proba = meta_learner.predict_proba(meta_features)
        result.stacked_proba[name] = _align_proba(raw_proba, meta_classes, NUM_CLASSES)

    return result


def scatter_stage_proba(windows_df, result: StageDetectorResult) -> np.ndarray:
    """Reassemble the (leakage-free: OOF for train, held-out for val/test)
    stacked stage probabilities into a single [N, NUM_CLASSES] array aligned
    to `windows_df`'s row order — this is what the impact forecaster
    (src/training/impact_forecast_trainer.py) consumes as "Stage 1/2
    evidence" for every window, regardless of split."""
    split = windows_df["split"].to_numpy()
    out = np.zeros((len(windows_df), NUM_CLASSES), dtype=np.float32)
    for name in ("train", "val", "test"):
        mask = split == name
        if mask.sum() == 0:
            continue
        out[mask] = result.stacked_proba[name]
    return out
