"""Generic config-driven dataset adapter.

Both ToN-IoT and Edge-IIoTset are flow/packet-level tabular captures with a
timestamp, a source/destination host pair, and an attack-type label — just
under different column names. Rather than duplicating the same loading
logic twice, this class does all the schema-matching / normalization work
driven entirely by a dataset's YAML config, and the two thin dataset-
specific subclasses (src/data/toniot_adapter.py,
src/data/edgeiiotset_adapter.py) only override the handful of things that
are genuinely dataset-specific: timestamp parsing quirks and (for
Edge-IIoTset) the impact-forecasting data-quality gate.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.base_adapter import load_concat_csvs, match_column, match_columns, resolve_raw_files
from src.data.schema import (
    ASSET_ID_COL,
    BINARY_LABEL_COL,
    DatasetBundle,
    PEER_ID_COL,
    RAW_ATTACK_TYPE_COL,
    TIMESTAMP_COL,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class GenericAdapter:
    """Base class for a config-driven dataset adapter."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_name = config["dataset"]["name"]

    # -- hooks a subclass may override -----------------------------------
    def parse_timestamp(self, raw: pd.Series, fmt: str) -> pd.Series:
        """Default timestamp parsing. `fmt` comes from schema.timestamp.format."""
        if fmt == "epoch_seconds":
            numeric = pd.to_numeric(raw, errors="coerce")
            return pd.to_datetime(numeric, unit="s", errors="coerce")
        # 'auto' or anything else: let pandas infer (handles Wireshark-style
        # strings like "Apr 26, 2021 11:16:12.123456000").
        return pd.to_datetime(raw.astype(str), errors="coerce")

    def extra_quality_checks(self, bundle: DatasetBundle) -> None:
        """Hook for dataset-specific quality gating (e.g. Edge-IIoTset's
        asset-cardinality / timestamp-parseability gate for forecasting).
        Mutates bundle.quality / bundle.warnings in place."""
        return

    # -- shared loading logic ---------------------------------------------
    def load(self) -> DatasetBundle:
        ds_cfg = self.config["dataset"]
        schema_cfg = self.config["schema"]

        files = resolve_raw_files(ds_cfg["raw_dir"], ds_cfg["preferred_filename"], ds_cfg["file_glob"])
        raw_df = load_concat_csvs(files)
        raw_df.columns = [c.strip() for c in raw_df.columns]

        warnings: List[str] = []
        matched = match_columns(
            raw_df.columns,
            {
                "timestamp": schema_cfg["timestamp"]["candidates"],
                "src_id": schema_cfg["src_id"]["candidates"],
                "dst_id": schema_cfg["dst_id"]["candidates"],
                "binary_label": schema_cfg["binary_label"]["candidates"],
                "attack_type": schema_cfg["attack_type"]["candidates"],
            },
        )
        logger.info(f"[{self.dataset_name}] Matched identity/label columns: {matched}")

        # --- hard requirements -------------------------------------------------
        if matched["timestamp"] is None:
            raise ValueError(
                f"[{self.dataset_name}] Could not find a timestamp column among "
                f"candidates {schema_cfg['timestamp']['candidates']} in header "
                f"{list(raw_df.columns)}. Refusing to fabricate one."
            )
        if matched["src_id"] is None:
            raise ValueError(
                f"[{self.dataset_name}] Could not find an asset/source-id column among "
                f"candidates {schema_cfg['src_id']['candidates']}. This field defines "
                f"the graph node identity and is required."
            )
        if matched["binary_label"] is None and matched["attack_type"] is None:
            raise ValueError(
                f"[{self.dataset_name}] Neither a binary label nor an attack-type "
                f"column was found; cannot supervise stage mapping."
            )

        out = pd.DataFrame(index=raw_df.index)
        out[TIMESTAMP_COL] = self.parse_timestamp(raw_df[matched["timestamp"]], schema_cfg["timestamp"]["format"])
        out[ASSET_ID_COL] = raw_df[matched["src_id"]].astype(str)

        has_peer_id = matched["dst_id"] is not None
        if has_peer_id:
            out[PEER_ID_COL] = raw_df[matched["dst_id"]].astype(str)
        else:
            out[PEER_ID_COL] = np.nan
            warnings.append(
                "No destination/peer-id column found: interaction edges cannot be "
                "built; graph construction will fall back to temporal-edges-only."
            )

        if matched["binary_label"] is not None:
            out[BINARY_LABEL_COL] = pd.to_numeric(raw_df[matched["binary_label"]], errors="coerce").fillna(0).astype(int)
        else:
            out[BINARY_LABEL_COL] = np.nan

        if matched["attack_type"] is not None:
            out[RAW_ATTACK_TYPE_COL] = raw_df[matched["attack_type"]].astype(str).str.strip()
            attack_type_source = "column"
        else:
            # Fall back to the binary label: 0 -> 'normal', 1 -> 'unknown_attack'.
            # We do NOT invent a specific attack family — downstream mapping
            # sends 'unknown_attack' through `default_unmapped`.
            out[RAW_ATTACK_TYPE_COL] = np.where(out[BINARY_LABEL_COL] == 0, "normal", "unknown_attack")
            attack_type_source = "binary_label_fallback"
            warnings.append(
                "No attack-type column found; derived a coarse benign/attack split "
                "from the binary label only. Fine-grained stage mapping will be less "
                "accurate for this dataset."
            )
        if matched["binary_label"] is None:
            # Derive binary label straight from the attack-type text using the
            # common 'normal'/'benign' convention. The authoritative benign/attack
            # split used for stage mapping still comes from stage_mapping_*.yaml
            # (src/mapping/label_mapper.py) — this is only a convenience column.
            out[BINARY_LABEL_COL] = (~out[RAW_ATTACK_TYPE_COL].str.strip().str.lower().isin(["normal", "benign"])).astype(int)

        # --- feature columns: only those actually present ----------------------
        numeric_candidates = schema_cfg.get("numeric_features", [])
        categorical_candidates = schema_cfg.get("categorical_features", [])
        drop_cols = set(schema_cfg.get("drop_cols", []))

        numeric_found = [c for c in numeric_candidates if c in raw_df.columns and c not in drop_cols]
        categorical_found = [c for c in categorical_candidates if c in raw_df.columns and c not in drop_cols]
        missing_numeric = sorted(set(numeric_candidates) - set(numeric_found) - drop_cols)
        missing_categorical = sorted(set(categorical_candidates) - set(categorical_found) - drop_cols)
        if missing_numeric:
            warnings.append(f"Numeric feature candidates not found in raw header (skipped): {missing_numeric}")
        if missing_categorical:
            warnings.append(f"Categorical feature candidates not found in raw header (skipped): {missing_categorical}")

        for c in numeric_found:
            out[c] = pd.to_numeric(raw_df[c], errors="coerce")
        for c in categorical_found:
            out[c] = raw_df[c].astype(str)

        # --- drop rows with unusable timestamp / asset id ------------------------
        n_before = len(out)
        bad_ts = out[TIMESTAMP_COL].isna()
        bad_asset = out[ASSET_ID_COL].isna() | (out[ASSET_ID_COL].str.lower().isin(["nan", "none", ""]))
        out = out.loc[~(bad_ts | bad_asset)].reset_index(drop=True)
        n_after = len(out)
        if n_after < n_before:
            warnings.append(
                f"Dropped {n_before - n_after:,} / {n_before:,} rows with unparseable "
                f"timestamp or missing asset id."
            )

        quality: Dict[str, Any] = {
            "n_rows_raw": int(n_before),
            "n_rows_kept": int(n_after),
            "n_distinct_assets": int(out[ASSET_ID_COL].nunique()),
            "timestamp_parseable_frac": float(1 - bad_ts.mean()) if n_before else 0.0,
            "has_peer_id": has_peer_id,
            "attack_type_source": attack_type_source,
        }

        bundle = DatasetBundle(
            df=out,
            numeric_feature_cols=numeric_found,
            categorical_feature_cols=categorical_found,
            has_peer_id=has_peer_id,
            dataset_name=self.dataset_name,
            matched_columns=matched,
            warnings=warnings,
            quality=quality,
        )
        self.extra_quality_checks(bundle)
        for w in bundle.warnings:
            logger.warning(f"[{self.dataset_name}] {w}")
        logger.info(f"[{self.dataset_name}] Data quality: {bundle.quality}")
        return bundle
