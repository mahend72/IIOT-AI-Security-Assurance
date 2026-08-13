"""Training loop for the GCN stage detector.

Reused both for (a) the FINAL GCN fit on the full train split (used for
val/test evaluation) and (b) each fold's GCN fit during out-of-fold
meta-feature generation (src/training/stage_detector_trainer.py) — in both
cases you pass in a `Data` object plus a `train_mask`/`stop_mask` and get a
trained model back. Which nodes are visible to the FORWARD pass (message
passing) vs. which are allowed to contribute to the LOSS is controlled
entirely by the `Data` object and masks the caller constructs — this
function never decides that on its own.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Data

from src.models.gcn_model import build_gcn
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _class_weights(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(y, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights


def train_gcn(
    data: Data,
    train_mask: torch.Tensor,
    stop_mask: torch.Tensor,
    num_classes: int,
    cfg: Dict[str, Any],
    seed: int = 42,
    verbose: bool = False,
    conv_type: str = "gcn",
) -> Tuple[torch.nn.Module, Dict[str, list]]:
    """train_mask: nodes used for the loss/gradient. stop_mask: nodes used ONLY
    to decide early stopping (never contributes gradient) — for the final
    model this is the true val split; for an OOF fold this is that fold's
    held-out nodes (documented design choice, see stage_detector_trainer.py).
    `conv_type`: 'gcn' | 'sage' | 'gat' -- selects the GNN architecture
    (src/models/gcn_model.py); default 'gcn' preserves prior behavior."""
    torch.manual_seed(seed)
    model = build_gcn(in_dim=data.x.shape[1], num_classes=num_classes, cfg=cfg, conv_type=conv_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 0.005), weight_decay=cfg.get("weight_decay", 5e-4))
    weights = _class_weights(data.y[train_mask], num_classes)

    epochs = cfg.get("epochs", 100)
    patience = cfg.get("patience", 15)
    best_score = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history = {"train_loss": [], "stop_macro_f1": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask], weight=weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            preds = logits.argmax(dim=-1)
            stop_f1 = f1_score(
                data.y[stop_mask].cpu().numpy(), preds[stop_mask].cpu().numpy(), average="macro", zero_division=0
            )
        history["train_loss"].append(float(loss.item()))
        history["stop_macro_f1"].append(float(stop_f1))
        if verbose and epoch % 10 == 0:
            logger.info(f"  epoch {epoch:03d} | loss {loss.item():.4f} | stop macro-F1 {stop_f1:.4f}")

        if stop_f1 > best_score:
            best_score = stop_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, history


@torch.no_grad()
def gcn_predict_proba(model: torch.nn.Module, data: Data) -> np.ndarray:
    model.eval()
    proba = model.predict_proba(data.x, data.edge_index)
    return proba.cpu().numpy()
