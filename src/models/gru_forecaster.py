"""Dual-stream GRU impact forecaster.

Per the paper spec: predicts whether an asset reaches IMP within horizon H,
using ONLY pre-impact Stage 1 (IAD) / Stage 2 (LMEP) evidence.

Architecture: each pre-impact window in an asset's timeline gets a raw
tabular feature vector plus the upstream stage detector's predicted
probability of that window being IAD / LMEP (renormalized over
{Benign, IAD, LMEP} — the IMP dimension is dropped entirely, never fed in,
which is the actual mechanism enforcing "no IMP evidence" at the feature
level, on top of the sequence-construction-level guarantee in
src/training/sequence_builder.py that no IMP-labeled window ever appears in
an input sequence in the first place).

Two parallel input channels are built per timestep:
    stream_IAD[t]  = P(IAD | window t)  * raw_features[t]   (+ P(IAD|t) itself)
    stream_LMEP[t] = P(LMEP | window t) * raw_features[t]   (+ P(LMEP|t) itself)
i.e. the same underlying evidence, gated by how much the stage detector
attributes it to each of the two pre-impact stages. Each channel is
encoded by its OWN GRU ("dual-stream"); the final hidden states are
concatenated and passed to a small MLP head producing P(reach IMP within H).
"""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class DualStreamGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru_iad = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=gru_dropout)
        self.gru_lmep = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=gru_dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x_iad: torch.Tensor,
        x_lmep: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """x_iad, x_lmep: [B, T, input_dim] (right-padded). lengths: [B] true
        sequence lengths. Returns raw logits [B] (apply sigmoid for P(IMP within H))."""
        lengths_cpu = lengths.detach().cpu()

        packed_iad = pack_padded_sequence(x_iad, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, h_iad = self.gru_iad(packed_iad)
        h_iad_last = h_iad[-1]  # [B, hidden_dim] — final layer's last hidden state

        packed_lmep = pack_padded_sequence(x_lmep, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, h_lmep = self.gru_lmep(packed_lmep)
        h_lmep_last = h_lmep[-1]

        combined = torch.cat([h_iad_last, h_lmep_last], dim=-1)
        logits = self.head(combined).squeeze(-1)
        return logits


def build_gru_forecaster(input_dim: int, cfg: Dict[str, Any]) -> DualStreamGRU:
    return DualStreamGRU(
        input_dim=input_dim,
        hidden_dim=cfg.get("hidden_dim", 32),
        num_layers=cfg.get("num_layers", 1),
        dropout=cfg.get("dropout", 0.2),
    )
