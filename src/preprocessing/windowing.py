"""Entity-time-window instance construction.

Builds the fundamental unit of this pipeline: an (asset, time-window)
instance. Every record is bucketed into a window using a GLOBAL time axis
(not per-asset-relative), which is essential — two different assets'
windows must line up in absolute time for "communication within the same
window" (interaction edges, src/graph/) to mean anything.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.schema import ASSET_ID_COL, PEER_ID_COL, TIMESTAMP_COL
from src.mapping.label_mapper import STAGE_SEVERITY

WINDOW_ID_COL = "window_id"
STAGE_LABEL_COL = "stage_label"
_SEVERITY_TO_STAGE = {v: k for k, v in STAGE_SEVERITY.items()}


def _fast_group_mode(work: pd.DataFrame, group_cols: List[str], col: str) -> pd.Series:
    """Vectorized per-group mode (most frequent value), avoiding a Python
    callable invoked once per group (which dominated runtime — `.agg(lambda
    s: s.mode()...)` over ~14k groups x 14 categorical columns was the
    single biggest cost in this function before this rewrite).

    Ties broken by whichever value groupby-size-sort happens to keep first,
    which is fine here: we only use the mode as a coarse categorical
    summary feature for a window, not as ground truth.
    """
    counts = work.groupby(group_cols + [col], sort=False).size().rename("__cnt").reset_index()
    counts = counts.sort_values("__cnt", ascending=False)
    top = counts.drop_duplicates(subset=group_cols, keep="first")
    return top.set_index(group_cols)[col]


def assign_window_id(df: pd.DataFrame, delta_t_seconds: float) -> pd.Series:
    """Global window bucket index: floor(unix_seconds / delta_t)."""
    epoch_seconds = df[TIMESTAMP_COL].astype("int64") / 1e9  # ns -> s
    return np.floor(epoch_seconds / delta_t_seconds).astype("int64")


def build_asset_window_instances(
    df: pd.DataFrame,
    delta_t_seconds: float,
    numeric_feature_cols: List[str],
    categorical_feature_cols: List[str],
    label_rule: str = "max_severity",
    window_id_override: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate raw (already stage-mapped) records into asset-window instances.

    Requires `df` to already have a STAGE_LABEL_COL column (see
    src/mapping/label_mapper.py) produced upstream, and a peer id column
    (may be all-NaN if the dataset lacks one).

    `window_id_override`: if given (aligned to `df`'s index), used verbatim
    as WINDOW_ID_COL instead of computing it from the timestamp column via
    `delta_t_seconds`. Used by the ASSET-LEVEL fallback (see
    `build_asset_level_instances` below) for datasets whose timestamp field
    cannot be trusted for real chronological bucketing (e.g. Edge-IIoTset —
    see `src/data/edgeiiotset_adapter.py`): passing a constant 0 collapses
    every one of an asset's records into a single aggregated instance, with
    no dependency on the (untrustworthy) timestamp value at all.

    Returns:
        windows_df: one row per (asset_id, window_id), with aggregated
            numeric features (mean/std/max/sum), the mode of each
            categorical feature, record count, and a window-level stage
            label.
        interactions_df: (asset_id, peer_id, window_id) rows — the raw
            material for interaction-edge construction in src/graph. Rows
            with a missing peer id are excluded (dataset has no peer field).
    """
    if label_rule != "max_severity":
        raise NotImplementedError(f"Unsupported label_rule '{label_rule}'")

    work = df.copy()
    if window_id_override is not None:
        work[WINDOW_ID_COL] = np.asarray(window_id_override)
    else:
        work[WINDOW_ID_COL] = assign_window_id(work, delta_t_seconds)

    group_cols = [ASSET_ID_COL, WINDOW_ID_COL]
    grouped = work.groupby(group_cols, sort=False)

    # A single grouped .agg() call for all built-in (C-vectorized) reductions
    # — mean/std/max/sum/nunique — reuses one internal group index across
    # every column instead of the O(n_features) separate groupby passes a
    # naive column-by-column loop would trigger. The two genuinely
    # per-group-Python-callable operations (categorical mode, stage-label
    # aggregation) are handled separately below via fully vectorized
    # groupby tricks instead of `.agg(lambda ...)`, which is what actually
    # dominated runtime before this rewrite (~14k groups x 14 categorical
    # columns of `.mode()` calls).
    agg_dict: Dict[str, list] = {TIMESTAMP_COL: ["min"]}
    for c in numeric_feature_cols:
        agg_dict[c] = ["mean", "std", "max", "sum"]
    for c in categorical_feature_cols:
        agg_dict[c] = ["nunique"]

    stats = grouped.agg(agg_dict)
    stats.columns = [
        "window_start_time" if col == (TIMESTAMP_COL, "min") else f"{col[0]}__{col[1]}"
        for col in stats.columns
    ]
    std_cols = [f"{c}__std" for c in numeric_feature_cols]
    stats[std_cols] = stats[std_cols].fillna(0.0)

    mode_frames = [_fast_group_mode(work, group_cols, c).rename(f"{c}__mode") for c in categorical_feature_cols]

    n_records = grouped.size().rename("n_records")

    # Window-level stage label: worst (highest-severity) stage among the
    # records in the window — computed via a vectorized severity-int max
    # instead of a per-group Python callable.
    # .astype(int) guards against STAGE_LABEL_COL being a pandas Categorical
    # dtype (as produced by LabelMapper.map_series) — .map() on a Categorical
    # can itself come back Categorical, and pandas categoricals refuse
    # groupby .max() unless explicitly ordered.
    severity = work[STAGE_LABEL_COL].map(STAGE_SEVERITY).astype("int64")
    max_severity = severity.groupby([work[ASSET_ID_COL], work[WINDOW_ID_COL]], sort=False).max()
    stage_series = max_severity.map(_SEVERITY_TO_STAGE).rename(STAGE_LABEL_COL)

    windows_df = pd.concat([n_records, stats, *mode_frames, stage_series], axis=1).reset_index()
    # .copy() defragments the frame (it was assembled via a wide `pd.concat`
    # of many single-column Series) so later single-column assignments
    # (e.g. pipeline.py's `windows_df["split"] = ...`) don't trigger
    # pandas' PerformanceWarning about a fragmented DataFrame.
    windows_df = windows_df.sort_values([ASSET_ID_COL, WINDOW_ID_COL]).reset_index(drop=True).copy()

    has_peer = PEER_ID_COL in work.columns and work[PEER_ID_COL].notna().any()
    if has_peer:
        interactions_df = (
            work.loc[work[PEER_ID_COL].notna(), [ASSET_ID_COL, PEER_ID_COL, WINDOW_ID_COL]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
    else:
        interactions_df = pd.DataFrame(columns=[ASSET_ID_COL, PEER_ID_COL, WINDOW_ID_COL])

    return windows_df, interactions_df


def build_record_level_instances(
    df: pd.DataFrame,
    numeric_feature_cols: List[str],
    categorical_feature_cols: List[str],
    label_rule: str = "max_severity",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """RECORD-LEVEL fallback: one instance per raw record (NOT aggregated
    across an asset's history), used instead of `build_asset_window_instances`
    when a dataset's timestamp field cannot be trusted for real chronological
    Delta-t bucketing (see `bundle.quality['timestamp_reliable_for_windowing']`,
    set by the dataset adapter; consumed by `src/pipeline.py`).

    Why record-level and not one-node-per-asset (aggregated over its whole
    history, taking the worst/max-severity stage ever seen): on Edge-IIoTset,
    high-volume-attack traffic (e.g. DDoS) is spread across a very large
    number of distinct source IPs that each contribute only a handful of
    records, so a "worst stage ever, per asset" aggregation collapses ~99.9%
    of assets straight to the IMP class -- a degenerate classification task
    with almost no Benign/IAD/LMEP instances left. Keeping one instance per
    RECORD preserves the dataset's real, natural class distribution (still
    with an asset-disjoint train/val/test split, so no leakage), at the cost
    of the "one node per asset" framing the main (ToN-IoT, real Delta-t
    windows) pipeline uses.

    Implemented as `build_asset_window_instances` with a globally unique
    `window_id_override` (one distinct id per row) so no two records of the
    same asset are ever aggregated together, and every record becomes its
    own node. Because these per-row ids carry NO verified chronological
    meaning (see the adapter's docstring), the caller (`src/pipeline.py`)
    MUST explicitly discard any temporal edges `graph_builder` derives from
    them rather than trust row-order-adjacency as a time proxy -- this
    function does not do that itself, since it only builds instances, not
    the graph.
    """
    window_id_override = pd.Series(np.arange(len(df)), index=df.index)
    return build_asset_window_instances(
        df, delta_t_seconds=float("nan"), numeric_feature_cols=numeric_feature_cols,
        categorical_feature_cols=categorical_feature_cols, label_rule=label_rule,
        window_id_override=window_id_override,
    )


def aggregated_feature_columns(windows_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Split windows_df columns into numeric vs categorical aggregate features
    (used by the tabular feature builder / graph node-feature builder)."""
    numeric_cols = [c for c in windows_df.columns if c.endswith(("__mean", "__std", "__max", "__sum", "__nunique"))]
    numeric_cols += ["n_records"] if "n_records" in windows_df.columns else []
    categorical_cols = [c for c in windows_df.columns if c.endswith("__mode")]
    return {"numeric": numeric_cols, "categorical": categorical_cols}
