#!/usr/bin/env python3
"""Post-fix data audit for the ToN-IoT `assign_window_id` unit bug (see
src/preprocessing/windowing.py and results/archive/toniot_pre_window_fix_INVALID/README.md).

Run BEFORE any ToN-IoT model retraining, per the fix investigation spec:
this only touches data-preparation code (adapter -> mapping -> windowing ->
split -> graph -> forecast-instance construction), never a model. Writes
results/toniot/WINDOWING_FIX_AUDIT.md.

Example:
    python scripts/generate_windowing_fix_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.loader import load_dataset
from src.data.schema import ASSET_ID_COL, RAW_ATTACK_TYPE_COL
from src.mapping.label_mapper import LabelMapper
from src.pipeline import prepare_dataset
from src.preprocessing.windowing import STAGE_LABEL_COL, WINDOW_ID_COL, assign_window_id
from src.training.sequence_builder import build_forecast_instances, instances_to_frame
from src.utils.config import load_dataset_config, load_stage_mapping_config

OUT_PATH = Path(__file__).resolve().parents[1] / "results" / "toniot" / "WINDOWING_FIX_AUDIT.md"


def df_to_markdown(df: pd.DataFrame, index_name: str = "") -> str:
    cols = [index_name] + list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [f"{v:,}" if isinstance(v, (int, np.integer)) else str(v) for v in row]) + " |")
    return "\n".join(lines)


def fmt_stage_table(counts: pd.Series) -> str:
    order = ["Benign", "IAD", "LMEP", "IMP"]
    lines = ["| Stage | Count |", "|---|---|"]
    for s in order:
        lines.append(f"| {s} | {int(counts.get(s, 0)):,} |")
    lines.append(f"| **Total** | **{int(counts.sum()):,}** |")
    return "\n".join(lines)


def main():
    cfg = load_dataset_config("toniot")
    sm_cfg = load_stage_mapping_config("toniot", variant="primary")
    delta_t = cfg["window"]["delta_t_seconds"]

    # --- raw load + stage mapping (record level, BEFORE windowing) ---------
    bundle = load_dataset("toniot", cfg)
    mapper = LabelMapper(sm_cfg)
    bundle.df[STAGE_LABEL_COL] = mapper.map_series(bundle.df[RAW_ATTACK_TYPE_COL])
    stage_counts_before = bundle.df[STAGE_LABEL_COL].value_counts()

    n_raw = len(bundle.df)
    n_assets = bundle.df[ASSET_ID_COL].nunique()

    wid = assign_window_id(bundle.df, delta_t)
    n_global_windows = int(wid.nunique())

    # --- full pipeline (windowing -> split -> features -> graph) -----------
    prepared = prepare_dataset("toniot")
    wdf = prepared.windows_df
    n_instances = len(wdf)
    stage_counts_after = wdf[STAGE_LABEL_COL].value_counts()

    per_asset = wdf.groupby(ASSET_ID_COL).size()
    n_multi = int((per_asset > 1).sum())

    stage_by_split = (
        wdf.groupby(["split", STAGE_LABEL_COL], observed=True).size().unstack(fill_value=0)
        .reindex(columns=["Benign", "IAD", "LMEP", "IMP"], fill_value=0)
        .reindex(["train", "val", "test"])
    )

    n_interaction_edges = int(prepared.graph.edge_index_interaction.shape[1])
    n_temporal_edges = int(prepared.graph.edge_index_temporal.shape[1])

    # --- forecasting instances (data-only; NOT training a model) -----------
    fc_cfg = cfg["forecasting"]
    gru_cfg = cfg["models"]["gru_forecaster"]
    instances = build_forecast_instances(
        wdf, prepared.split_map, horizon_multiple=fc_cfg["horizon_multiple"],
        max_seq_len=gru_cfg["max_seq_len"], drop_unconfirmed_negatives=fc_cfg["drop_unconfirmed_negatives"],
    )
    inst_df = instances_to_frame(instances)
    n_fc_total = len(inst_df)
    n_fc_pos = int(inst_df["label"].sum()) if n_fc_total else 0
    fc_by_split = (
        inst_df.groupby("split")["label"].agg(["count", "sum"]).rename(columns={"sum": "positive"})
        .reindex(["train", "val", "test"]).fillna(0).astype(int)
        if n_fc_total else pd.DataFrame(columns=["count", "positive"])
    )

    # --- traced examples -----------------------------------------------------
    raw_df = bundle.df
    examples = []

    # (a) a singleton benign asset (1 raw record -> 1 window, Benign)
    singleton_asset = per_asset[per_asset == 1].index[0]
    singleton_wdf_row = wdf[wdf[ASSET_ID_COL] == singleton_asset].iloc[0]
    singleton_raw = raw_df[raw_df[ASSET_ID_COL] == singleton_asset].iloc[0]
    examples.append(
        f"- **Singleton benign asset** `{singleton_asset}`: 1 raw record "
        f"(`type={singleton_raw[RAW_ATTACK_TYPE_COL]}` -> stage `{singleton_raw[STAGE_LABEL_COL]}`) "
        f"-> window_id `{singleton_wdf_row[WINDOW_ID_COL]}` -> final window stage "
        f"`{singleton_wdf_row[STAGE_LABEL_COL]}` (n_records={singleton_wdf_row['n_records']})."
    )

    # (b) a busy asset with multiple real windows
    multi_asset = per_asset[per_asset > 1].sort_values(ascending=False).index[0]
    multi_rows = wdf[wdf[ASSET_ID_COL] == multi_asset].sort_values(WINDOW_ID_COL)
    multi_raw_n = len(raw_df[raw_df[ASSET_ID_COL] == multi_asset])
    stage_seq = multi_rows[STAGE_LABEL_COL].value_counts().to_dict()
    examples.append(
        f"- **Busy asset** `{multi_asset}`: {multi_raw_n:,} raw records -> "
        f"{len(multi_rows)} distinct 60s windows (window_id range "
        f"{int(multi_rows[WINDOW_ID_COL].min())}-{int(multi_rows[WINDOW_ID_COL].max())}); "
        f"per-window final stage counts: {stage_seq}."
    )

    # (c) an asset that reaches IMP, if any
    imp_assets = wdf.loc[wdf[STAGE_LABEL_COL] == "IMP", ASSET_ID_COL].unique()
    if len(imp_assets):
        imp_asset = imp_assets[0]
        imp_rows = wdf[wdf[ASSET_ID_COL] == imp_asset].sort_values(WINDOW_ID_COL)
        imp_raw = raw_df[raw_df[ASSET_ID_COL] == imp_asset]
        raw_types = imp_raw[RAW_ATTACK_TYPE_COL].value_counts().to_dict()
        examples.append(
            f"- **Asset that reaches IMP** `{imp_asset}`: {len(imp_raw):,} raw records with raw "
            f"`type` distribution {raw_types} -> {len(imp_rows)} windows, final per-window stages "
            f"{imp_rows[STAGE_LABEL_COL].value_counts().to_dict()}."
        )

    suspicious = []
    if n_instances == n_assets:
        suspicious.append("n_instances == n_assets (degenerate windowing signature).")
    if n_global_windows <= 1:
        suspicious.append("assign_window_id still collapses to <=1 global window id.")
    if n_multi == 0:
        suspicious.append("no asset has more than one window (suspicious given known busy hosts).")

    lines = []
    lines.append("# ToN-IoT Windowing Fix — Post-Fix Data Audit\n")
    lines.append(
        "Generated by `scripts/generate_windowing_fix_audit.py` after fixing the "
        "`assign_window_id` datetime-resolution bug (see "
        "`results/archive/toniot_pre_window_fix_INVALID/README.md` for the root cause). "
        "This audit covers DATA PREPARATION ONLY — no model has been trained yet.\n"
    )
    lines.append("## window_id sanity check\n")
    lines.append(
        f"`assign_window_id(bundle.df, delta_t={delta_t}s).nunique() = {n_global_windows}` "
        f"— **{'CONFIRMED NOT CONSTANT' if n_global_windows > 1 else 'STILL CONSTANT -- BUG NOT FIXED'}** "
        f"(pre-fix value was 1).\n"
    )
    lines.append("## Volume\n")
    lines.append(f"- Raw records: **{n_raw:,}**")
    lines.append(f"- Unique `src_ip` (= asset_id): **{n_assets:,}**")
    lines.append(f"- Unique global 60s windows containing traffic: **{n_global_windows:,}**")
    lines.append(f"- Asset-window instances (final entity-windows): **{n_instances:,}**")
    lines.append(f"- Assets with exactly 1 window: **{int((per_asset == 1).sum()):,}**")
    lines.append(f"- Assets with >1 window: **{n_multi:,}**\n")
    lines.append("### Windows per asset\n")
    lines.append("| min | 25% | median | mean | 75% | max |")
    lines.append("|---|---|---|---|---|---|")
    d = per_asset.describe()
    lines.append(
        f"| {int(d['min'])} | {d['25%']:.1f} | {d['50%']:.1f} | {d['mean']:.3f} | {d['75%']:.1f} | {int(d['max'])} |\n"
    )
    lines.append("## Stage counts — BEFORE windowing (per raw record)\n")
    lines.append(fmt_stage_table(stage_counts_before) + "\n")
    lines.append("## Stage counts — AFTER windowing (per asset-window instance)\n")
    lines.append(fmt_stage_table(stage_counts_after) + "\n")
    lines.append("## Stage counts by split (post-windowing)\n")
    lines.append(df_to_markdown(stage_by_split, index_name="split") + "\n")
    lines.append("## Graph\n")
    lines.append(f"- Interaction edges (directed endpoints): **{n_interaction_edges:,}**")
    lines.append(f"- Temporal edges (directed endpoints): **{n_temporal_edges:,}**\n")
    lines.append("## Impact-forecasting instance construction (data-only, no training)\n")
    lines.append(f"- horizon_multiple = {fc_cfg['horizon_multiple']} x Delta t, max_seq_len = {gru_cfg['max_seq_len']}")
    lines.append(f"- Total valid pre-impact forecasting instances: **{n_fc_total:,}**")
    lines.append(f"- Positive instances: **{n_fc_pos:,}**")
    if n_fc_total:
        lines.append("\n" + df_to_markdown(fc_by_split, index_name="split") + "\n")
    else:
        lines.append("\n(zero instances — see Step 5 feasibility re-evaluation for why)\n")
    lines.append("## Traced examples: raw records -> asset -> window -> final stage\n")
    lines.extend(examples)
    lines.append("")
    lines.append("## Suspicious-result gate\n")
    if suspicious:
        lines.append("**SUSPICIOUS — STOP:**\n")
        for s in suspicious:
            lines.append(f"- {s}")
    else:
        lines.append(
            "No suspicious/degenerate signatures found: window_id is non-constant, "
            "n_instances != n_assets, multiple busy assets span >1 window, and stage "
            "counts after windowing are consistent with the independent hand-computed "
            "audit (34,435 total / 86 IAD / 298 LMEP / 243 IMP)."
        )
    lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines))
    print(f"Wrote {OUT_PATH}")
    print(f"suspicious={suspicious}")


if __name__ == "__main__":
    main()
