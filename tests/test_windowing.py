"""Regression tests for src/preprocessing/windowing.py::assign_window_id.

Added after a confirmed critical bug: `assign_window_id` assumed
`df[TIMESTAMP_COL]` was always pandas `datetime64[ns]` and divided
`.astype("int64")` by 1e9 unconditionally. Under pandas 3.0.1,
`pd.to_datetime(numeric, unit="s")` returns `datetime64[s]` for the ToN-IoT
timestamp column, so `.astype("int64")` already returned whole seconds --
dividing by 1e9 again crushed every ToN-IoT timestamp to a value between
1.554 and 1.557, giving `floor(x / 60) == 0` for all 461,043 records.

These tests pin the resolution-independent behavior so this cannot silently
regress again on a future pandas/numpy upgrade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.schema import TIMESTAMP_COL
from src.preprocessing.windowing import assign_window_id

# Epoch seconds 0, 59, 60, 119, 120 with delta_t=60 -> window ids 0, 0, 1, 1, 2.
KNOWN_EPOCH_SECONDS = [0, 59, 60, 119, 120]
EXPECTED_WINDOW_IDS = [0, 0, 1, 1, 2]

RESOLUTIONS = ["s", "ms", "us", "ns"]


@pytest.mark.parametrize("unit", RESOLUTIONS)
def test_known_timestamps_all_resolutions(unit):
    """Same epoch instants, represented at four different datetime64
    resolutions, must all produce the same window ids."""
    ts = pd.to_datetime(KNOWN_EPOCH_SECONDS, unit="s").astype(f"datetime64[{unit}]")
    assert ts.dtype == np.dtype(f"datetime64[{unit}]")
    df = pd.DataFrame({TIMESTAMP_COL: ts})

    result = assign_window_id(df, delta_t_seconds=60)

    assert result.tolist() == EXPECTED_WINDOW_IDS
    assert result.dtype == np.int64


def test_all_resolutions_agree_with_each_other():
    """Cross-check: results for [s]/[ms]/[us]/[ns] representations of the
    SAME instants must be identical to one another, not just individually
    correct against the hand-computed expectation above."""
    results = {}
    for unit in RESOLUTIONS:
        ts = pd.to_datetime(KNOWN_EPOCH_SECONDS, unit="s").astype(f"datetime64[{unit}]")
        df = pd.DataFrame({TIMESTAMP_COL: ts})
        results[unit] = assign_window_id(df, delta_t_seconds=60).tolist()

    values = list(results.values())
    assert all(v == values[0] for v in values), results


def test_window_boundary_is_half_open():
    """floor(x/60): the boundary second (60, 120, ...) starts the NEXT
    window, not the previous one."""
    ts = pd.to_datetime([59.999, 60.0, 60.001], unit="s")
    df = pd.DataFrame({TIMESTAMP_COL: ts})
    result = assign_window_id(df, delta_t_seconds=60)
    assert result.tolist() == [0, 1, 1]


def test_does_not_silently_collapse_a_wide_real_world_range():
    """Regression guard for the actual failure mode: a realistic ToN-IoT-like
    epoch-second range (~1.55 billion, spanning >1 hour) must NOT collapse
    to a single window id the way the buggy `/1e9`-on-already-seconds
    version did (it mapped every such timestamp to window id 0)."""
    epoch_seconds = np.linspace(1_554_198_358, 1_554_198_358 + 7200, 1000)  # 2 hours
    ts = pd.to_datetime(epoch_seconds, unit="s")
    df = pd.DataFrame({TIMESTAMP_COL: ts})
    result = assign_window_id(df, delta_t_seconds=60)
    assert result.nunique() > 1
    # ~2 hours / 60s = ~120 distinct windows expected.
    assert result.nunique() >= 100


def test_rejects_unparseable_timestamp():
    df = pd.DataFrame({TIMESTAMP_COL: ["not a timestamp", "also not one"]})
    with pytest.raises(Exception):
        assign_window_id(df, delta_t_seconds=60)
