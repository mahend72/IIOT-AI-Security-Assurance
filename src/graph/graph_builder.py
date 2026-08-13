"""Asset-time interaction graph construction.

  nodes            = asset-window instances (one row of `windows_df`)
  interaction edges = two assets communicated within the same window
                       (derived from `interactions_df`, i.e. src/dst pairs
                       observed in that window — both endpoints must
                       themselves be monitored assets with a node in that
                       window; edges to a peer with no node of its own in
                       that window cannot be represented as a graph edge)
  temporal edges     = consecutive OBSERVED windows for the same asset
                       (i.e. an asset's windows in time order, linked
                       window[t] -> window[t+1]; windows an asset has no
                       traffic in simply have no node, so "consecutive"
                       means consecutive among the asset's own observed
                       windows, not consecutive integer window ids)

Both edge sets are built once and cached; the graph-ablation experiment
(interaction-only / temporal-only / both) just chooses which to include in
`edge_index` when materializing a PyG `Data` object — no need to rebuild
from raw records for each ablation arm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data.schema import ASSET_ID_COL, PEER_ID_COL
from src.mapping.label_mapper import STAGE_ORDER
from src.preprocessing.windowing import WINDOW_ID_COL
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

GraphMode = Literal["interaction_only", "temporal_only", "both"]
STAGE_TO_INT = {s: i for i, s in enumerate(STAGE_ORDER)}


@dataclass
class AssetTimeGraph:
    """Holds both edge sets + node bookkeeping so any ablation mode can be
    materialized into a PyG Data object without recomputation."""

    node_table: pd.DataFrame  # windows_df with an added contiguous 'node_id' column
    edge_index_interaction: torch.Tensor  # [2, E_int], both directions included
    edge_index_temporal: torch.Tensor  # [2, E_temp], both directions included

    def edge_index_for_mode(self, mode: GraphMode) -> torch.Tensor:
        if mode == "interaction_only":
            return self.edge_index_interaction
        if mode == "temporal_only":
            return self.edge_index_temporal
        if mode == "both":
            if self.edge_index_interaction.numel() == 0:
                return self.edge_index_temporal
            if self.edge_index_temporal.numel() == 0:
                return self.edge_index_interaction
            return torch.cat([self.edge_index_interaction, self.edge_index_temporal], dim=1)
        raise ValueError(f"Unknown graph mode '{mode}'")

    def to_pyg_data(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mode: GraphMode = "both",
    ) -> Data:
        edge_index = self.edge_index_for_mode(mode)
        data = Data(
            x=torch.as_tensor(np.array(x, dtype=np.float32, copy=True)),
            edge_index=edge_index,
            y=torch.as_tensor(np.array(y, dtype=np.int64, copy=True)),
        )
        for split in ("train", "val", "test"):
            mask = torch.as_tensor((self.node_table["split"] == split).to_numpy().copy(), dtype=torch.bool)
            data[f"{split}_mask"] = mask
        return data

    def induced_subgraph(
        self,
        x: np.ndarray,
        y: np.ndarray,
        keep_row_mask: np.ndarray,
        mode: GraphMode = "both",
    ) -> Tuple[Data, np.ndarray]:
        """Return a PyG Data object containing ONLY the nodes where
        `keep_row_mask` is True, with edges re-indexed to the new local
        node numbering (edges touching a dropped node are dropped too).

        Used to build a TRAIN-ONLY subgraph for out-of-fold meta-feature
        generation (src/training/stage_detector_trainer.py), so that GCN
        message passing during OOF fold training can never touch a val/test
        node's features — not just its labels.

        Returns (sub_data, global_row_indices) where global_row_indices[i]
        is the row index into the original node_table / x / y that local
        node i corresponds to (needed to scatter fold predictions back).
        """
        keep_row_mask = np.asarray(keep_row_mask, dtype=bool)
        global_indices = np.nonzero(keep_row_mask)[0]
        remap = -np.ones(len(keep_row_mask), dtype=np.int64)
        remap[global_indices] = np.arange(len(global_indices))

        def _remap_edges(edge_index: torch.Tensor) -> torch.Tensor:
            if edge_index.numel() == 0:
                return edge_index
            src, dst = edge_index.numpy()
            keep = keep_row_mask[src] & keep_row_mask[dst]
            new_src = remap[src[keep]]
            new_dst = remap[dst[keep]]
            return torch.as_tensor(np.stack([new_src, new_dst]), dtype=torch.long)

        sub_interaction = _remap_edges(self.edge_index_interaction)
        sub_temporal = _remap_edges(self.edge_index_temporal)

        if mode == "interaction_only":
            edge_index = sub_interaction
        elif mode == "temporal_only":
            edge_index = sub_temporal
        elif sub_interaction.numel() == 0:
            edge_index = sub_temporal
        elif sub_temporal.numel() == 0:
            edge_index = sub_interaction
        else:
            edge_index = torch.cat([sub_interaction, sub_temporal], dim=1)

        sub_data = Data(
            x=torch.as_tensor(np.array(x[global_indices], dtype=np.float32, copy=True)),
            edge_index=edge_index,
            y=torch.as_tensor(np.array(y[global_indices], dtype=np.int64, copy=True)),
        )
        return sub_data, global_indices


def _build_node_index(windows_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], int]]:
    node_table = windows_df.reset_index(drop=True).copy()
    node_table["node_id"] = node_table.index.values
    index = {
        (row[ASSET_ID_COL], row[WINDOW_ID_COL]): row["node_id"]
        for row in node_table[[ASSET_ID_COL, WINDOW_ID_COL, "node_id"]].to_dict("records")
    }
    return node_table, index


def _build_interaction_edges(
    node_table: pd.DataFrame,
    interactions_df: pd.DataFrame,
    node_index: Dict[Tuple[str, int], int],
    max_edges_per_window: int,
) -> torch.Tensor:
    """Vectorized construction: an interaction edge only exists between two
    nodes that BOTH have a window instance (i.e. both endpoints are
    monitored assets that originated traffic in that window) — a peer that
    never appears as an `asset_id` itself (e.g. an external server this
    dataset never observed as a source) cannot be represented as a node, so
    traffic to it cannot become a graph edge. This is done with two merges
    against the node table rather than a per-window Python loop, which is
    what makes this tractable on full-scale (100k+ row) datasets."""
    if interactions_df.empty:
        logger.warning("No interaction records available (dataset lacks a peer/destination id) — "
                        "interaction edge set will be empty.")
        return torch.empty((2, 0), dtype=torch.long)

    src_lookup = node_table[[ASSET_ID_COL, WINDOW_ID_COL, "node_id"]]
    dst_lookup = src_lookup.rename(columns={ASSET_ID_COL: PEER_ID_COL, "node_id": "peer_node_id"})

    merged = interactions_df.merge(src_lookup, on=[ASSET_ID_COL, WINDOW_ID_COL], how="inner")
    merged = merged.merge(dst_lookup, on=[PEER_ID_COL, WINDOW_ID_COL], how="inner")
    merged = merged.loc[merged["node_id"] != merged["peer_node_id"], ["node_id", "peer_node_id", WINDOW_ID_COL]]

    if merged.empty:
        logger.warning(
            "Interaction records exist, but none connect two assets that BOTH have a node "
            "in that window (peers are all non-monitored hosts) — interaction edge set is empty."
        )
        return torch.empty((2, 0), dtype=torch.long)

    # Cap edges per window to bound graph size on high-fanout windows (e.g. a
    # scan touching thousands of destinations). Only the (rare) oversized
    # windows are looped over.
    counts = merged.groupby(WINDOW_ID_COL).size()
    oversized = counts[counts > max_edges_per_window].index
    if len(oversized) > 0:
        rng = np.random.default_rng(0)
        keep = [merged[~merged[WINDOW_ID_COL].isin(oversized)]]
        for w in oversized:
            sub = merged[merged[WINDOW_ID_COL] == w]
            keep.append(sub.sample(n=max_edges_per_window, random_state=int(rng.integers(0, 1_000_000))))
        merged = pd.concat(keep, ignore_index=True)

    edges = merged[["node_id", "peer_node_id"]].drop_duplicates().to_numpy(dtype=np.int64).T
    edges = np.concatenate([edges, edges[::-1]], axis=1)  # make undirected
    edges = np.unique(edges, axis=1)
    return torch.as_tensor(edges, dtype=torch.long)


def _build_temporal_edges(node_table: pd.DataFrame) -> torch.Tensor:
    src_list, dst_list = [], []
    for asset_id, group in node_table.sort_values(WINDOW_ID_COL).groupby(ASSET_ID_COL):
        node_ids = group["node_id"].values
        if len(node_ids) < 2:
            continue
        src_list.extend(node_ids[:-1])
        dst_list.extend(node_ids[1:])

    if not src_list:
        return torch.empty((2, 0), dtype=torch.long)

    edges = np.array([src_list, dst_list], dtype=np.int64)
    edges = np.concatenate([edges, edges[::-1]], axis=1)  # make undirected
    edges = np.unique(edges, axis=1)
    return torch.as_tensor(edges, dtype=torch.long)


def build_asset_time_graph(
    windows_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    asset_split_map: Dict[str, str],
    max_interaction_edges_per_window: int = 200_000,
) -> AssetTimeGraph:
    node_table, node_index = _build_node_index(windows_df)
    node_table["split"] = node_table[ASSET_ID_COL].map(asset_split_map)
    if node_table["split"].isna().any():
        missing = node_table.loc[node_table["split"].isna(), ASSET_ID_COL].unique()
        raise ValueError(f"{len(missing)} assets have no split assignment (e.g. {missing[:5]}). "
                          f"Every asset must be assigned by src.preprocessing.splitting.asset_disjoint_split first.")

    edge_index_interaction = _build_interaction_edges(node_table, interactions_df, node_index, max_interaction_edges_per_window)
    edge_index_temporal = _build_temporal_edges(node_table)

    logger.info(
        f"Graph built: {len(node_table):,} nodes | "
        f"{edge_index_interaction.shape[1]:,} interaction-edge endpoints | "
        f"{edge_index_temporal.shape[1]:,} temporal-edge endpoints"
    )
    return AssetTimeGraph(node_table=node_table, edge_index_interaction=edge_index_interaction, edge_index_temporal=edge_index_temporal)
