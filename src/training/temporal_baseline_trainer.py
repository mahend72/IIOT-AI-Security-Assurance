"""Trainer for the no-graph temporal (GRU/LSTM) stage-detection baseline.

Builds, for every (asset, window) instance, the causal prefix of that
asset's own OBSERVED windows up to and including the instance itself
(chronological order by WINDOW_ID_COL, truncated to `max_seq_len`), and
trains a single-stream RNN (src/models/temporal_stage_model.py) to predict
the instance's own stage label from that prefix. Requires a dataset whose
timestamp/window ordering is trustworthy (see
`PreparedDataset.used_asset_level_fallback` -- callers should not invoke
this for a dataset where that flag is True; e.g. run_stage_detection
documents this as N/A for edgeiiotset instead of silently training on an
untrustworthy row order)."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.schema import ASSET_ID_COL
from src.models.temporal_stage_model import build_temporal_stage_classifier
from src.preprocessing.windowing import WINDOW_ID_COL
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TemporalSeqInstance:
    row_index: int  # positional index into windows_df / X / y this instance predicts
    split: str
    prefix_row_indices: List[int]


def build_temporal_sequences(windows_df: pd.DataFrame, max_seq_len: int) -> List[TemporalSeqInstance]:
    instances: List[TemporalSeqInstance] = []
    for asset_id, group in windows_df.groupby(ASSET_ID_COL, sort=False):
        group = group.sort_values(WINDOW_ID_COL)
        global_indices = group.index.tolist()
        splits = group["split"].tolist()
        for pos in range(len(global_indices)):
            start = max(0, pos - max_seq_len + 1)
            prefix = global_indices[start : pos + 1]
            instances.append(TemporalSeqInstance(row_index=global_indices[pos], split=splits[pos], prefix_row_indices=prefix))
    return instances


@dataclass
class TemporalBaselineResult:
    model: Any = None
    y_true: Dict[str, np.ndarray] = field(default_factory=dict)
    proba: Dict[str, np.ndarray] = field(default_factory=dict)
    history: Dict[str, list] = field(default_factory=dict)


def train_temporal_baseline(
    windows_df: pd.DataFrame, X: np.ndarray, y: np.ndarray, num_classes: int, cfg: Dict[str, Any], seed: int = 42,
) -> TemporalBaselineResult:
    torch.manual_seed(seed)
    rnn_cfg = cfg["models"].get("temporal_baseline", {"hidden_dim": 32, "num_layers": 1, "dropout": 0.2,
                                                       "cell": "gru", "lr": 1e-3, "weight_decay": 1e-4,
                                                       "epochs": 60, "patience": 10, "batch_size": 64, "max_seq_len": 60})
    max_seq_len = rnn_cfg.get("max_seq_len", 60)
    instances = build_temporal_sequences(windows_df, max_seq_len)

    n = len(instances)
    d = X.shape[1]
    Xt = np.zeros((n, max_seq_len, d), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int64)
    labels = np.zeros(n, dtype=np.int64)
    splits = np.empty(n, dtype=object)
    row_indices = np.zeros(n, dtype=np.int64)
    for i, inst in enumerate(instances):
        idxs = inst.prefix_row_indices
        t = len(idxs)
        Xt[i, :t] = X[idxs]
        lengths[i] = t
        labels[i] = y[inst.row_index]
        splits[i] = inst.split
        row_indices[i] = inst.row_index

    result = TemporalBaselineResult()
    train_mask, val_mask, test_mask = splits == "train", splits == "val", splits == "test"

    device = torch.device("cpu")
    model = build_temporal_stage_classifier(input_dim=d, num_classes=num_classes, cfg=rnn_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=rnn_cfg.get("lr", 1e-3), weight_decay=rnn_cfg.get("weight_decay", 1e-4))

    counts = np.bincount(labels[train_mask], minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    class_weights = torch.as_tensor(counts.sum() / (num_classes * counts), dtype=torch.float32)

    def make_loader(mask, shuffle):
        if mask.sum() == 0:
            return None
        ds = TensorDataset(torch.from_numpy(Xt[mask]), torch.from_numpy(lengths[mask]), torch.from_numpy(labels[mask]))
        return DataLoader(ds, batch_size=rnn_cfg.get("batch_size", 64), shuffle=shuffle)

    train_loader = make_loader(train_mask, True)
    val_loader = make_loader(val_mask, False)

    epochs = rnn_cfg.get("epochs", 60)
    patience = rnn_cfg.get("patience", 10)
    best_val = np.inf
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        total, n_seen = 0.0, 0
        for xb, lb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb, lb)
            loss = torch.nn.functional.cross_entropy(logits, yb, weight=class_weights)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(yb)
            n_seen += len(yb)
        train_loss = total / max(n_seen, 1)

        if val_loader is not None:
            model.eval()
            vtotal, vn = 0.0, 0
            with torch.no_grad():
                for xb, lb, yb in val_loader:
                    logits = model(xb, lb)
                    loss = torch.nn.functional.cross_entropy(logits, yb, weight=class_weights)
                    vtotal += float(loss.item()) * len(yb)
                    vn += len(yb)
            val_loss = vtotal / max(vn, 1)
        else:
            val_loss = train_loss

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
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
                result.proba[name] = np.zeros((0, num_classes))
                continue
            logits = model(torch.from_numpy(Xt[mask]), torch.from_numpy(lengths[mask]))
            proba = torch.softmax(logits, dim=-1).numpy()
            result.y_true[name] = labels[mask]
            result.proba[name] = proba

    return result
