"""Normalized in-memory schema that every dataset adapter must produce.

Downstream code (mapping, windowing, graph construction, models) only ever
talks to this normalized schema, never to raw ToN-IoT / Edge-IIoTset column
names directly. That is what lets the same pipeline run on both datasets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

# Standardized column names produced by every adapter.
TIMESTAMP_COL = "timestamp"
ASSET_ID_COL = "asset_id"
PEER_ID_COL = "peer_id"
RAW_ATTACK_TYPE_COL = "raw_attack_type"
BINARY_LABEL_COL = "binary_label"


@dataclass
class DatasetBundle:
    """Output of a dataset adapter's `.load()`.

    Attributes:
        df: normalized dataframe with TIMESTAMP_COL, ASSET_ID_COL,
            (optionally) PEER_ID_COL, RAW_ATTACK_TYPE_COL, BINARY_LABEL_COL,
            plus raw numeric_feature_cols / categorical_feature_cols columns
            (kept under their ORIGINAL raw names — no renaming/fabrication).
        numeric_feature_cols: subset of numeric feature candidates that were
            actually found in this dataset's real header.
        categorical_feature_cols: same, for categorical candidates.
        has_peer_id: whether a destination/peer identifier column was found
            (needed for interaction edges).
        dataset_name: e.g. 'toniot' or 'edgeiiotset'.
        matched_columns: audit trail of {standard_field: raw_column_name}.
        warnings: human-readable list of things the adapter had to fall back
            on or could not find (surfaced in the data-quality report).
        quality: free-form dict of data-quality stats used to gate whether
            impact forecasting is valid for this dataset.
    """

    df: pd.DataFrame
    numeric_feature_cols: List[str]
    categorical_feature_cols: List[str]
    has_peer_id: bool
    dataset_name: str
    matched_columns: Dict[str, Optional[str]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    quality: Dict[str, object] = field(default_factory=dict)
