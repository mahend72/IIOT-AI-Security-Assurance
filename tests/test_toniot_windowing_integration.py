"""Integration regression tests: run the REAL ToN-IoT pipeline end-to-end
through the fixed `assign_window_id` and check the result against an
independent, hand-computed audit of `data/raw/toniot/Train_Test_Network.csv`
(see the trace performed when this bug was found).

These are slow (load + process a 461k-row / 70MB CSV) and require the raw
ToN-IoT file to be present locally, so they are skipped rather than failed
when the file is absent (e.g. in a CI environment without the dataset).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.schema import ASSET_ID_COL
from src.pipeline import prepare_dataset
from src.preprocessing.windowing import WINDOW_ID_COL, assign_window_id
from src.utils.config import load_dataset_config
from src.data.loader import load_dataset

RAW_TONIOT_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "toniot" / "Train_Test_Network.csv"

pytestmark = pytest.mark.skipif(
    not RAW_TONIOT_CSV.exists(),
    reason=f"raw ToN-IoT CSV not found at {RAW_TONIOT_CSV} -- integration test requires the real dataset.",
)


@pytest.fixture(scope="module")
def prepared():
    return prepare_dataset("toniot")


def test_window_id_is_not_constant_on_real_data():
    """The original bug's direct symptom: window_id must vary across the
    dataset's real ~27-day timestamp range, not collapse to a single value."""
    cfg = load_dataset_config("toniot")
    bundle = load_dataset("toniot", cfg)
    wid = assign_window_id(bundle.df, cfg["window"]["delta_t_seconds"])
    assert wid.nunique() > 1, (
        "assign_window_id collapsed to a single window id across the whole dataset -- "
        "this is the exact symptom of the ns/1e9 unit bug."
    )
    # Independent audit: 4,100 distinct 60s buckets actually contain traffic
    # over the dataset's ~27-day span (traffic is bursty, not continuous).
    assert wid.nunique() == 4100, f"expected 4,100 distinct global window ids, got {wid.nunique()}"


def test_entity_window_counts_match_independent_audit(prepared):
    """Compare the full pipeline's windows_df against the independent audit
    (hand-computed groupby(['src_ip', floor(ts/60)]).max(severity) over the
    raw CSV, done separately from `build_asset_window_instances`).

    Per the investigation spec: do NOT force these values -- if the real
    pipeline disagrees, this test must fail loudly and explain why, not
    silently pass with a loosened tolerance.
    """
    wdf = prepared.windows_df
    counts = wdf["stage_label"].value_counts().to_dict()

    assert len(wdf) == 34435, f"expected 34,435 total asset-window instances, got {len(wdf)}"
    assert counts.get("IAD", 0) == 86, f"expected 86 IAD windows, got {counts.get('IAD', 0)}"
    assert counts.get("LMEP", 0) == 298, f"expected 298 LMEP windows, got {counts.get('LMEP', 0)}"
    assert counts.get("IMP", 0) == 243, f"expected 243 IMP windows, got {counts.get('IMP', 0)}"
    assert counts.get("Benign", 0) == 33808, f"expected 33,808 Benign windows, got {counts.get('Benign', 0)}"


def test_final_instances_not_degenerately_equal_to_asset_count(prepared):
    """Post-fix, the number of (asset, window) instances must differ from
    the number of distinct assets -- unless that equality genuinely follows
    from the data (every asset really does fit in one window). Here it must
    NOT hold, because we know from the raw audit that 84 assets have >1
    window (heavy hosts whose traffic spans multiple real 60s buckets)."""
    n_windows = len(prepared.windows_df)
    n_assets = prepared.bundle.df[ASSET_ID_COL].nunique()
    assert n_windows != n_assets, (
        f"n_windows ({n_windows}) == n_assets ({n_assets}) again -- this is the exact "
        f"degenerate signature of the window_id-constant bug. Investigate before proceeding."
    )
    assert n_windows > n_assets  # real windowing must SPLIT some assets into multiple instances


def test_multiple_busy_assets_have_more_than_one_window(prepared):
    per_asset = prepared.windows_df.groupby(ASSET_ID_COL).size()
    n_multi = int((per_asset > 1).sum())
    assert n_multi == 84, f"expected 84 assets with >1 window, got {n_multi}"
    assert per_asset.max() > 1


def test_window_start_time_is_chronological_within_each_asset(prepared):
    """For a given asset, window_start_time must be non-decreasing as
    window_id increases -- i.e. window ids really do track real elapsed
    time (required by sequence_builder's pre-impact-history construction),
    not an arbitrary/scrambled bucket index."""
    wdf = prepared.windows_df.sort_values([ASSET_ID_COL, WINDOW_ID_COL])
    non_monotonic = (
        wdf.groupby(ASSET_ID_COL)["window_start_time"]
        .apply(lambda s: (s.diff().dropna() < pd.Timedelta(0)).any())
    )
    assert not non_monotonic.any(), (
        f"{int(non_monotonic.sum())} asset(s) have non-chronological window_start_time "
        f"as window_id increases: {non_monotonic[non_monotonic].index.tolist()[:10]}"
    )


def test_no_asset_split_overlap(prepared):
    """An asset must appear in exactly one of train/val/test -- never split
    across windows of the same asset (record-level leakage)."""
    splits_per_asset = prepared.windows_df.groupby(ASSET_ID_COL)["split"].nunique()
    offenders = splits_per_asset[splits_per_asset > 1]
    assert offenders.empty, f"{len(offenders)} asset(s) span multiple splits: {offenders.index.tolist()[:10]}"


def test_feature_scaler_fit_on_train_only(prepared):
    """The tabular feature builder's scaler/encoder must have been fit only
    on TRAIN-split windows (leakage check), per src/pipeline.py step 4."""
    fb = prepared.feature_builder
    n_train_windows = int((prepared.windows_df["split"] == "train").sum())
    # StandardScaler exposes n_samples_seen_ after fit; it must equal the
    # TRAIN-split row count, not the full (train+val+test) window count.
    scaler = fb.column_transformer.named_transformers_["num"].named_steps["scale"]
    seen = int(np.atleast_1d(scaler.n_samples_seen_)[0])
    assert seen == n_train_windows, (
        f"feature scaler was fit on {seen} rows, expected exactly the "
        f"{n_train_windows} TRAIN-split window rows -- possible leakage."
    )
    assert seen != len(prepared.windows_df)  # sanity: must not equal the FULL dataset
