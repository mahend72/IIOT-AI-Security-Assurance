"""Dataset summary table (results/<dataset>/main/dataset_summary.csv):
raw records (by modality), final unified instances, feature dimension,
class/stage distribution, graph nodes/edges/avg degree, train/val/test
split counts, impact-forecasting positive/negative instance counts (or the
documented reason they are not applicable)."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.data.schema import ASSET_ID_COL
from src.evaluation.forecast_validity import check_instance_level_validity
from src.training.sequence_builder import build_forecast_instances


def build_dataset_summary_rows(prepared) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def add(metric, value, note=""):
        rows.append({"metric": metric, "value": value, "note": note})

    q = prepared.bundle.quality
    add("raw_records_modality", "network_flow (single CSV source)")
    add("raw_records_count", q.get("n_rows_raw"))
    add("raw_records_kept_after_adapter_cleaning", q.get("n_rows_kept"))
    add(
        "unit_of_analysis",
        "asset-time-window (Delta t seconds)" if not prepared.used_asset_level_fallback else "record-level (no trustworthy timestamp for windowing)",
    )
    add("final_unified_instances", len(prepared.windows_df))
    add("feature_dimension", prepared.X.shape[1])
    add("delta_t_seconds", prepared.delta_t_seconds if not prepared.used_asset_level_fallback else float("nan"),
        "" if not prepared.used_asset_level_fallback else "N/A -- record-level fallback, see data_quality_report")
    add("n_distinct_assets", prepared.windows_df[ASSET_ID_COL].nunique())

    stage_counts = prepared.windows_df["stage_label"].value_counts()
    for stage in ["Benign", "IAD", "LMEP", "IMP"]:
        add(f"stage_count_{stage}", int(stage_counts.get(stage, 0)))

    for split in ("train", "val", "test"):
        add(f"split_count_{split}", int((prepared.windows_df["split"] == split).sum()))

    add("graph_nodes", len(prepared.graph.node_table))
    n_interaction_endpoints = int(prepared.graph.edge_index_interaction.shape[1])
    n_temporal_endpoints = int(prepared.graph.edge_index_temporal.shape[1])
    add("graph_interaction_edges", n_interaction_endpoints // 2, "undirected edges (endpoints / 2)")
    add("graph_temporal_edges", n_temporal_endpoints // 2, "undirected edges (endpoints / 2)")
    n_nodes = len(prepared.graph.node_table)
    total_degree_endpoints = n_interaction_endpoints + n_temporal_endpoints
    avg_degree = total_degree_endpoints / n_nodes if n_nodes else float("nan")
    add("graph_average_degree", round(avg_degree, 4), "(interaction+temporal edge endpoints) / nodes")

    n_impact_assets = prepared.windows_df.loc[prepared.windows_df["stage_label"] == "IMP", ASSET_ID_COL].nunique()
    n_total_assets = prepared.windows_df[ASSET_ID_COL].nunique()
    add("impact_positive_asset_count", int(n_impact_assets))
    add("impact_negative_asset_count", int(n_total_assets - n_impact_assets))

    metadata_gate_valid = q.get("impact_forecasting_valid", True)
    if metadata_gate_valid:
        fc_cfg = prepared.cfg["forecasting"]
        gru_cfg = prepared.cfg["models"]["gru_forecaster"]
        instances = build_forecast_instances(
            prepared.windows_df, prepared.split_map, horizon_multiple=fc_cfg["horizon_multiple"],
            max_seq_len=gru_cfg["max_seq_len"], drop_unconfirmed_negatives=fc_cfg["drop_unconfirmed_negatives"],
        )
        instance_level_valid, instance_reasons = check_instance_level_validity(instances)
        labels = np.array([i.label for i in instances]) if instances else np.array([])
        splits_arr = np.array([i.split for i in instances]) if instances else np.array([])
        add("impact_forecast_instances_total", len(instances))
        add("impact_forecast_positive_total", int(labels.sum()) if len(labels) else 0)
        add("impact_forecast_negative_total", int((labels == 0).sum()) if len(labels) else 0)
        for split in ("train", "val", "test"):
            mask = splits_arr == split
            add(f"impact_forecast_positive_{split}", int(labels[mask].sum()) if mask.sum() else 0)
            add(f"impact_forecast_negative_{split}", int((labels[mask] == 0).sum()) if mask.sum() else 0)
        add("impact_forecasting_valid_overall", instance_level_valid,
            "" if instance_level_valid else "; ".join(instance_reasons))
    else:
        add("impact_forecast_instances_total", float("nan"), "N/A -- " + "; ".join(q.get("impact_forecasting_invalid_reasons", [])))
        add("impact_forecasting_valid_overall", False, "; ".join(q.get("impact_forecasting_invalid_reasons", [])))

    return rows


def dataset_summary_dataframe(prepared) -> pd.DataFrame:
    return pd.DataFrame(build_dataset_summary_rows(prepared))
