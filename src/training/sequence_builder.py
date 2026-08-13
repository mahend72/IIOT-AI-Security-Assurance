"""Builds pre-impact forecasting instances from window-level data.

CRITICAL LEAKAGE RULE (project spec): the impact forecaster may only see
Stage 1 (IAD) / Stage 2 (LMEP) evidence — never IMP evidence, and never
evidence "from the future" relative to the point it is making a prediction
from.

We enforce this structurally, not just by feature selection:
  - For every asset, we find the position of its FIRST IMP-labeled window
    (if any). Every candidate "cut point" (the window a forecast is made
    from) must be strictly BEFORE that position, so an IMP-labeled window
    can never appear anywhere in a training/inference input sequence.
  - The label ("does this asset reach IMP within horizon H windows after
    the cut point") is computed purely from each asset's own timeline, and
    is only accepted as a *confirmed* negative if we actually observed at
    least H windows after the cut point with no impact — otherwise
    (recording simply ends too soon to know) the instance is dropped
    rather than guessed, per `drop_unconfirmed_negatives`.
  - `window_id` (a GLOBAL Delta-t bucket index, see src/preprocessing/
    windowing.py) is used for all horizon-gap arithmetic, not positional
    index, so the horizon is measured in true elapsed time even when an
    asset has gaps in its own observed traffic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.schema import ASSET_ID_COL
from src.preprocessing.windowing import STAGE_LABEL_COL, WINDOW_ID_COL


@dataclass
class ForecastInstance:
    asset_id: str
    split: str
    label: int
    prefix_row_indices: List[int]  # positional indices into windows_df, chronological order, length <= max_seq_len
    cut_window_id: int


def build_forecast_instances(
    windows_df: pd.DataFrame,
    asset_split_map: Dict[str, str],
    horizon_multiple: int,
    max_seq_len: int,
    drop_unconfirmed_negatives: bool = True,
) -> List[ForecastInstance]:
    """windows_df must already be sorted by [asset_id, window_id] (true of the
    output of src.preprocessing.windowing.build_asset_window_instances)."""
    instances: List[ForecastInstance] = []
    n_dropped_unconfirmed = 0

    for asset_id, group in windows_df.groupby(ASSET_ID_COL, sort=False):
        stages = group[STAGE_LABEL_COL].tolist()
        window_ids = group[WINDOW_ID_COL].tolist()
        global_indices = group.index.tolist()  # positional row index into windows_df
        split = asset_split_map[asset_id]

        first_imp_pos: Optional[int] = next((i for i, s in enumerate(stages) if s == "IMP"), None)
        eligible_positions = range(first_imp_pos) if first_imp_pos is not None else range(len(stages))

        for pos in eligible_positions:
            if first_imp_pos is not None:
                gap = window_ids[first_imp_pos] - window_ids[pos]
                label = int(gap <= horizon_multiple)
            else:
                gap_to_end = window_ids[-1] - window_ids[pos]
                if gap_to_end >= horizon_multiple:
                    label = 0
                else:
                    n_dropped_unconfirmed += 1
                    if drop_unconfirmed_negatives:
                        continue
                    label = 0  # only reached if caller explicitly disabled dropping

            start = max(0, pos - max_seq_len + 1)
            prefix_positions = list(range(start, pos + 1))
            instances.append(
                ForecastInstance(
                    asset_id=asset_id,
                    split=split,
                    label=label,
                    prefix_row_indices=[global_indices[p] for p in prefix_positions],
                    cut_window_id=window_ids[pos],
                )
            )

    return instances


def instances_to_frame(instances: List[ForecastInstance]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset_id": [i.asset_id for i in instances],
            "split": [i.split for i in instances],
            "label": [i.label for i in instances],
            "seq_len": [len(i.prefix_row_indices) for i in instances],
            "cut_window_id": [i.cut_window_id for i in instances],
        }
    )
