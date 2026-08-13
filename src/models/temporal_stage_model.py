"""No-graph temporal baseline for stage detection: a single GRU or LSTM
consumes an asset's own window-feature history (causal prefix ending at the
window being classified) and predicts that window's stage from the final
hidden state. No cross-asset message passing at all (contrast with the GCN/
GraphSAGE/GAT graph-based detectors) -- isolates how much of the graph
models' performance comes from temporal self-history alone vs. from
neighbor information."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TemporalStageClassifier(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 1,
        dropout: float = 0.2, cell: str = "gru",
    ):
        super().__init__()
        rnn_dropout = dropout if num_layers > 1 else 0.0
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.cell = cell
        self.rnn = rnn_cls(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=rnn_dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x: [B, T, input_dim] right-padded. lengths: [B]. Returns class
        logits [B, num_classes] from the final valid timestep's hidden state."""
        lengths_cpu = lengths.detach().cpu()
        packed = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        out = self.rnn(packed)
        h = out[1][0] if self.cell == "lstm" else out[1]
        h_last = h[-1]  # [B, hidden_dim] final layer's last hidden state
        return self.head(h_last)


def build_temporal_stage_classifier(input_dim: int, num_classes: int, cfg: Dict[str, Any]) -> TemporalStageClassifier:
    return TemporalStageClassifier(
        input_dim=input_dim,
        hidden_dim=cfg.get("hidden_dim", 32),
        num_classes=num_classes,
        num_layers=cfg.get("num_layers", 1),
        dropout=cfg.get("dropout", 0.2),
        cell=cfg.get("cell", "gru"),
    )
