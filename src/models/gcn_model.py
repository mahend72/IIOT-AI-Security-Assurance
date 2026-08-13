"""Graph neural network stage-detector — the graph-aware base learner
feeding the stacking meta-learner. Operates on the asset-time interaction
graph built by src/graph/graph_builder.py (node = asset-window instance).

Generalized beyond plain GCN to also support GraphSAGE (`SAGEConv`) and GAT
(`GATConv`) via `conv_type`, so the reviewer/manuscript model-comparison
table (RF / GCN / GraphSAGE / GAT / ...) can reuse the exact same training
loop (src/training/gcn_trainer.py) for all three graph architectures —
only the conv layer construction differs; message-passing interface
(`forward(x, edge_index) -> logits`) is identical."""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

_CONV_BUILDERS = {
    "gcn": lambda in_d, out_d: GCNConv(in_d, out_d),
    "sage": lambda in_d, out_d: SAGEConv(in_d, out_d),
    # GAT concatenates head outputs; use a single head so `out_d` is exact
    # (keeps in/out plumbing identical to GCN/SAGE for the shared trainer).
    "gat": lambda in_d, out_d: GATConv(in_d, out_d, heads=1, concat=True),
}


class GCN(nn.Module):
    """Despite the class name (kept for backward compatibility with
    existing imports/checkpoints), this is a generic message-passing GNN
    whose conv layer type is selected by `conv_type` ('gcn' | 'sage' |
    'gat')."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        conv_type: str = "gcn",
    ):
        super().__init__()
        assert num_layers >= 1
        if conv_type not in _CONV_BUILDERS:
            raise ValueError(f"Unknown conv_type '{conv_type}'. Known: {list(_CONV_BUILDERS)}")
        self.conv_type = conv_type
        self.dropout = dropout
        build = _CONV_BUILDERS[conv_type]
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(build(in_dim, num_classes))
        else:
            self.convs.append(build(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(build(hidden_dim, hidden_dim))
            self.convs.append(build(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Returns raw class logits, shape [N, num_classes]."""
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(x, edge_index)
        return F.softmax(logits, dim=-1)


def build_gcn(in_dim: int, num_classes: int, cfg: Dict[str, Any], conv_type: str = "gcn") -> GCN:
    return GCN(
        in_dim=in_dim,
        hidden_dim=cfg.get("hidden_dim", 64),
        num_classes=num_classes,
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.3),
        conv_type=conv_type,
    )
