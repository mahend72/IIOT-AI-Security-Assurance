"""Impact forecaster training: dual-stream GRU over pre-impact evidence.

Pipeline:
  1. Take the (leakage-free) per-window stage probabilities produced by the
     stage detector — OOF for train rows, held-out for val/test rows
     (src/training/stage_detector_trainer.py `.stacked_proba`) — and drop
     the IMP probability entirely, renormalizing over {Benign, IAD, LMEP}.
     This is the concrete mechanism that prevents IMP evidence from ever
     reaching the forecaster, on top of the sequence-construction-level
     guarantee (src/training/sequence_builder.py) that no window at/after
     an asset's first IMP-labeled window is ever included in an input
     sequence.
  2. Build two evidence streams per window: IAD-probability-gated features
     and LMEP-probability-gated features.
  3. Assemble fixed-length (padded) prefix sequences per forecasting
     instance (src/training/sequence_builder.py) and train the
     DualStreamGRU (src/models/gru_forecaster.py) to predict "reaches IMP
     within horizon H".
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.gru_forecaster import build_gru_forecaster
from src.training.sequence_builder import ForecastInstance
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_row_stream_features(X: np.ndarray, stage_proba_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """stage_proba_full: [N, 4] probabilities in STAGE_ORDER = [Benign, IAD, LMEP, IMP].
    Returns (feat_iad, feat_lmep), each [N, 1 + X.shape[1]] = [gate_prob, gate_prob * raw_features]."""
    eps = 1e-8
    proba3 = stage_proba_full[:, :3]  # drop IMP column entirely -- never touches the forecaster
    proba3 = proba3 / (proba3.sum(axis=1, keepdims=True) + eps)
    p_iad = proba3[:, 1:2]
    p_lmep = proba3[:, 2:3]
    feat_iad = np.concatenate([p_iad, X * p_iad], axis=1).astype(np.float32)
    feat_lmep = np.concatenate([p_lmep, X * p_lmep], axis=1).astype(np.float32)
    return feat_iad, feat_lmep


def instances_to_tensors(
    instances: List[ForecastInstance], feat_iad_full: np.ndarray, feat_lmep_full: np.ndarray, max_seq_len: int
):
    n = len(instances)
    d = feat_iad_full.shape[1]
    X_iad = np.zeros((n, max_seq_len, d), dtype=np.float32)
    X_lmep = np.zeros((n, max_seq_len, d), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int64)
    labels = np.zeros(n, dtype=np.int64)
    splits = np.empty(n, dtype=object)
    asset_ids = np.empty(n, dtype=object)
    cut_window_ids = np.zeros(n, dtype=np.int64)

    for i, inst in enumerate(instances):
        idxs = inst.prefix_row_indices
        t = len(idxs)
        X_iad[i, :t] = feat_iad_full[idxs]
        X_lmep[i, :t] = feat_lmep_full[idxs]
        lengths[i] = t
        labels[i] = inst.label
        splits[i] = inst.split
        asset_ids[i] = inst.asset_id
        cut_window_ids[i] = inst.cut_window_id

    return X_iad, X_lmep, lengths, labels, splits, asset_ids, cut_window_ids


@dataclass
class ForecastResult:
    model: Any = None
    y_true: Dict[str, np.ndarray] = field(default_factory=dict)
    y_proba: Dict[str, np.ndarray] = field(default_factory=dict)
    asset_ids: Dict[str, np.ndarray] = field(default_factory=dict)
    cut_window_ids: Dict[str, np.ndarray] = field(default_factory=dict)
    history: Dict[str, list] = field(default_factory=dict)
    n_instances: Dict[str, int] = field(default_factory=dict)
    positive_rate: Dict[str, float] = field(default_factory=dict)


def _run_epoch(model, loader, device, optimizer=None, pos_weight=None):
    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0
    all_logits, all_labels = [], []
    for x_iad, x_lmep, lengths, labels in loader:
        x_iad, x_lmep, lengths, labels = x_iad.to(device), x_lmep.to(device), lengths.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()
        logits = model(x_iad, x_lmep, lengths)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels.float(), pos_weight=pos_weight
        )
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        n += len(labels)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
    return total_loss / max(n, 1), torch.cat(all_logits), torch.cat(all_labels)


def train_impact_forecaster(
    instances: List[ForecastInstance],
    feat_iad_full: np.ndarray,
    feat_lmep_full: np.ndarray,
    cfg: Dict[str, Any],
    seed: int = 42,
) -> ForecastResult:
    torch.manual_seed(seed)
    gru_cfg = cfg["models"]["gru_forecaster"]
    max_seq_len = gru_cfg.get("max_seq_len", 60)

    X_iad, X_lmep, lengths, labels, splits, asset_ids, cut_window_ids = instances_to_tensors(
        instances, feat_iad_full, feat_lmep_full, max_seq_len
    )
    result = ForecastResult()
    for name in ("train", "val", "test"):
        mask = splits == name
        result.n_instances[name] = int(mask.sum())
        result.positive_rate[name] = float(labels[mask].mean()) if mask.sum() else float("nan")
    logger.info(f"Forecasting instances: {result.n_instances}, positive rates: {result.positive_rate}")

    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"
    if train_mask.sum() == 0:
        raise ValueError("No training instances for impact forecaster (check horizon/window config vs. data span).")

    device = torch.device("cpu")
    model = build_gru_forecaster(input_dim=X_iad.shape[-1], cfg=gru_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=gru_cfg.get("lr", 1e-3), weight_decay=gru_cfg.get("weight_decay", 1e-4))

    def make_loader(mask, shuffle):
        if mask.sum() == 0:
            return None
        ds = TensorDataset(
            torch.from_numpy(X_iad[mask]), torch.from_numpy(X_lmep[mask]),
            torch.from_numpy(lengths[mask]), torch.from_numpy(labels[mask]),
        )
        return DataLoader(ds, batch_size=gru_cfg.get("batch_size", 64), shuffle=shuffle)

    train_loader = make_loader(train_mask, shuffle=True)
    val_loader = make_loader(val_mask, shuffle=False)

    n_pos = labels[train_mask].sum()
    n_neg = train_mask.sum() - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    epochs = gru_cfg.get("epochs", 60)
    patience = gru_cfg.get("patience", 10)
    best_val_loss = np.inf
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        train_loss, _, _ = _run_epoch(model, train_loader, device, optimizer, pos_weight)
        if val_loader is not None:
            val_loss, _, _ = _run_epoch(model, val_loader, device, optimizer=None, pos_weight=pos_weight)
        else:
            val_loss = train_loss  # no val instances (tiny dataset) — fall back to train loss
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    result.model = model
    result.history = history

    with torch.no_grad():
        for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
            if mask.sum() == 0:
                result.y_true[name] = np.array([])
                result.y_proba[name] = np.array([])
                result.asset_ids[name] = np.array([])
                result.cut_window_ids[name] = np.array([])
                continue
            logits = model(
                torch.from_numpy(X_iad[mask]), torch.from_numpy(X_lmep[mask]), torch.from_numpy(lengths[mask])
            )
            proba = torch.sigmoid(logits).numpy()
            result.y_true[name] = labels[mask]
            result.y_proba[name] = proba
            result.asset_ids[name] = asset_ids[mask]
            result.cut_window_ids[name] = cut_window_ids[mask]

    return result
