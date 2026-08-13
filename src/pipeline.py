"""Shared orchestration: load -> map -> asset-disjoint split -> window ->
tabular features -> graph. Every script in scripts/ builds on
`prepare_dataset`, so the steps and their leakage-prevention rules
(feature scaling fit on TRAIN only, split decided before any of the
following steps, etc.) are defined exactly once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from src.data.loader import load_dataset
from src.data.schema import ASSET_ID_COL, PEER_ID_COL, RAW_ATTACK_TYPE_COL
from src.graph.graph_builder import AssetTimeGraph, STAGE_TO_INT, build_asset_time_graph
from src.mapping.label_mapper import LabelMapper
from src.preprocessing.features import TabularFeatureBuilder
from src.preprocessing.splitting import asset_disjoint_split
from src.preprocessing.windowing import (
    STAGE_LABEL_COL,
    aggregated_feature_columns,
    build_asset_window_instances,
    build_record_level_instances,
)
from src.utils.config import load_dataset_config, load_stage_mapping_config, override
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PreparedDataset:
    dataset_name: str
    cfg: Dict[str, Any]
    sm_cfg: Dict[str, Any]
    bundle: Any  # DatasetBundle
    windows_df: pd.DataFrame
    interactions_df: pd.DataFrame
    split_map: Dict[str, str]
    feature_builder: TabularFeatureBuilder
    X: np.ndarray
    y: np.ndarray
    graph: AssetTimeGraph
    delta_t_seconds: float
    mapping_variant: str = "primary"
    seed: int = 42
    used_asset_level_fallback: bool = False


def prepare_dataset(
    dataset_name: str,
    delta_t_seconds: Optional[float] = None,
    seed: Optional[int] = None,
    mapping_variant: str = "primary",
    preloaded_bundle: Optional[Any] = None,
) -> PreparedDataset:
    """Runs the full data->graph preparation pipeline for one dataset.

    `delta_t_seconds` / `seed` override the dataset config's defaults —
    used by the window-size sweep without needing a separate code path.
    `mapping_variant` selects an alternate stage_mapping_<dataset>_<variant>
    .yaml (see configs/ and run_reviewer_experiments.py's
    stage_mapping_sensitivity experiment) instead of the primary mapping.
    `preloaded_bundle` lets a caller that's sweeping several configs over
    the SAME raw data (e.g. the mapping-sensitivity or window-sensitivity
    experiments) skip re-reading and re-adapting the raw CSV every time --
    safe because everything this function derives from the bundle
    (stage labels, windows, features, graph) is recomputed fresh from it
    on every call regardless.
    """
    cfg = load_dataset_config(dataset_name)
    sm_cfg = load_stage_mapping_config(dataset_name, variant=mapping_variant)
    if delta_t_seconds is not None:
        cfg = override(cfg, "window.delta_t_seconds", delta_t_seconds)
    if seed is not None:
        cfg = override(cfg, "split.seed", seed)

    delta_t = cfg["window"]["delta_t_seconds"]
    effective_seed = cfg["split"]["seed"]
    logger.info(f"[{dataset_name}] Preparing dataset (Delta t = {delta_t}s, mapping='{mapping_variant}')...")

    bundle = preloaded_bundle if preloaded_bundle is not None else load_dataset(dataset_name, cfg)

    # 1. Map raw attack labels -> {Benign, IAD, LMEP, IMP}. Uses ONLY the
    # per-record raw attack-type string; no data-dependent statistic.
    mapper = LabelMapper(sm_cfg)
    bundle.df[STAGE_LABEL_COL] = mapper.map_series(bundle.df[RAW_ATTACK_TYPE_COL])
    if mapper.unmapped_values_seen:
        logger.warning(f"[{dataset_name}] Raw attack-type values not in stage_mapping config "
                        f"(fell back to default_unmapped): {mapper.unmapped_values_seen}")

    # 2. Asset-disjoint split, decided BEFORE any windowing/scaling/graph
    # step below can possibly depend on it.
    split_cfg = cfg["split"]
    split_map = asset_disjoint_split(
        bundle.df[ASSET_ID_COL].tolist(),
        split_cfg["train_frac"], split_cfg["val_frac"], split_cfg["test_frac"], split_cfg["seed"],
    )

    # 3. Entity-time-window instances -- or, if this dataset's timestamp field
    # is not reliable enough for real chronological Delta-t bucketing (see
    # `bundle.quality['timestamp_reliable_for_windowing']`, set by the
    # dataset adapter, e.g. src/data/edgeiiotset_adapter.py), fall back to
    # RECORD-LEVEL instances (no windowing/aggregation, no dependency on the
    # untrustworthy timestamp). This keeps stage detection + an
    # interaction-only graph valid while structurally invalidating anything
    # that needs real time (temporal edges, window-size sensitivity, impact
    # forecasting) -- those degrade to "no edges" / "no instances" rather
    # than silently using fabricated timestamps.
    used_asset_level_fallback = not bundle.quality.get("timestamp_reliable_for_windowing", True)
    if used_asset_level_fallback:
        logger.warning(
            f"[{dataset_name}] timestamp_reliable_for_windowing=False -- using RECORD-LEVEL "
            f"(non-windowed) instances instead of Delta-t={delta_t}s windowing."
        )
        windows_df, interactions_df = build_record_level_instances(
            bundle.df, bundle.numeric_feature_cols, bundle.categorical_feature_cols,
            label_rule=cfg["window"].get("label_rule", "max_severity"),
        )
    else:
        windows_df, interactions_df = build_asset_window_instances(
            bundle.df, delta_t, bundle.numeric_feature_cols, bundle.categorical_feature_cols,
            label_rule=cfg["window"].get("label_rule", "max_severity"),
        )
    windows_df["split"] = windows_df[ASSET_ID_COL].map(split_map)

    # 4. Tabular features -- scaler/encoder fit on TRAIN windows only.
    agg_cols = aggregated_feature_columns(windows_df)
    feature_builder = TabularFeatureBuilder(agg_cols["numeric"], agg_cols["categorical"])
    feature_builder.fit(windows_df.loc[windows_df["split"] == "train"])
    X = feature_builder.transform(windows_df)
    y = windows_df[STAGE_LABEL_COL].map(STAGE_TO_INT).to_numpy()

    # 5. Asset-time interaction graph (both edge types built; ablation mode
    # is chosen later, per-experiment, from this same graph).
    graph = build_asset_time_graph(
        windows_df, interactions_df, split_map, cfg["graph"]["max_interaction_edges_per_window"]
    )
    if used_asset_level_fallback:
        # RECORD-LEVEL fallback's window ids carry no verified chronological
        # meaning (they only exist to keep every record its own graph node --
        # see build_record_level_instances' docstring), so any "temporal"
        # edges graph_builder derived from row-order adjacency would be
        # fabricated, not real elapsed-time adjacency. Discard them
        # explicitly rather than let them silently pass through.
        n_dropped = graph.edge_index_temporal.shape[1]
        graph.edge_index_temporal = torch.empty((2, 0), dtype=torch.long)
        if n_dropped:
            logger.warning(
                f"[{dataset_name}] Discarded {n_dropped:,} row-order-derived pseudo-temporal "
                f"edge endpoints (no trustworthy timestamp for this dataset) -- "
                f"interaction-only graph."
            )
        # The standard "communicated in the SAME window" interaction-edge
        # definition (src/graph/graph_builder.py::_build_interaction_edges)
        # structurally cannot produce any edge at record granularity: every
        # record is its own uniquely-id'd "window" by construction (see
        # build_record_level_instances), so no two records ever share a
        # window id. Rebuild a coarser but still REAL (never fabricated)
        # interaction graph instead: connect one representative record-node
        # per asset to the representative node of every peer it was ever
        # observed communicating with, ignoring window/time entirely (there
        # is none to use). This is the only interaction-edge definition
        # that is both grounded strictly in observed src/dst pairs and
        # tractable at record granularity -- true record-level all-pairs
        # edges for high-fan-out hosts (e.g. a gateway with thousands of
        # peers) would be combinatorially intractable.
        rep_node = graph.node_table.groupby(ASSET_ID_COL)["node_id"].first()
        pairs = interactions_df[[ASSET_ID_COL, PEER_ID_COL]].drop_duplicates()
        pairs = pairs[pairs[PEER_ID_COL].isin(rep_node.index)]
        if len(pairs):
            src_nodes = pairs[ASSET_ID_COL].map(rep_node).to_numpy(dtype=np.int64)
            dst_nodes = pairs[PEER_ID_COL].map(rep_node).to_numpy(dtype=np.int64)
            edges = np.stack([src_nodes, dst_nodes])
            edges = edges[:, edges[0] != edges[1]]
            edges = np.concatenate([edges, edges[::-1]], axis=1)
            edges = np.unique(edges, axis=1) if edges.size else edges
            graph.edge_index_interaction = torch.as_tensor(edges, dtype=torch.long)
        else:
            graph.edge_index_interaction = torch.empty((2, 0), dtype=torch.long)
        logger.info(
            f"[{dataset_name}] Record-level fallback interaction graph: "
            f"{graph.edge_index_interaction.shape[1]:,} representative-node edge endpoints "
            f"(asset-pair communication, window/time-independent)."
        )

    logger.info(
        f"[{dataset_name}] Prepared {len(windows_df):,} window instances "
        f"({(windows_df['split'] == 'train').sum():,} train / "
        f"{(windows_df['split'] == 'val').sum():,} val / "
        f"{(windows_df['split'] == 'test').sum():,} test), "
        f"{X.shape[1]} features."
    )

    return PreparedDataset(
        dataset_name=dataset_name, cfg=cfg, sm_cfg=sm_cfg, bundle=bundle,
        windows_df=windows_df, interactions_df=interactions_df, split_map=split_map,
        feature_builder=feature_builder, X=X, y=y, graph=graph, delta_t_seconds=delta_t,
        mapping_variant=mapping_variant, seed=effective_seed,
        used_asset_level_fallback=used_asset_level_fallback,
    )
