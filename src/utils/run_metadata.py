"""Run-provenance metadata: dataset name, seed, config hash, UTC timestamp.

Validation check D ("all CSV/JSON outputs include dataset name, seed,
split, timestamp, and config hash") requires every result artifact to be
traceable back to exactly the config that produced it. Rather than bolt
this onto each script separately, every orchestration script builds one
`RunMetadata` per run and threads it through its JSON/CSV writers.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd


def config_hash(cfg: Dict[str, Any]) -> str:
    """Stable short hash of a config dict, independent of key order —
    two runs are comparable iff this hash matches."""
    canonical = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class RunMetadata:
    dataset: str
    seed: int
    config_hash: str
    timestamp_utc: str = field(default_factory=utc_timestamp)
    split: Optional[str] = None  # set per-row/per-artifact when one split is being described
    mapping_variant: str = "primary"
    is_real_data: bool = True
    data_source_note: str = ""

    @classmethod
    def build(
        cls,
        dataset: str,
        cfg: Dict[str, Any],
        seed: int,
        split: Optional[str] = None,
        mapping_variant: str = "primary",
        is_real_data: bool = True,
        data_source_note: str = "",
    ) -> "RunMetadata":
        return cls(
            dataset=dataset, seed=seed, config_hash=config_hash(cfg), split=split,
            mapping_variant=mapping_variant, is_real_data=is_real_data, data_source_note=data_source_note,
        )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def stamp_json(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of `obj` with a `_run_metadata` block attached."""
        out = dict(obj)
        out["_run_metadata"] = self.as_dict()
        return out

    def stamp_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of `df` with the metadata fields as constant columns
        (split, if set, overrides the df's own `split` column only if absent)."""
        df = df.copy()
        df["dataset"] = self.dataset
        df["seed"] = self.seed
        df["config_hash"] = self.config_hash
        df["timestamp_utc"] = self.timestamp_utc
        df["mapping_variant"] = self.mapping_variant
        df["is_real_data"] = self.is_real_data
        if self.split is not None and "split" not in df.columns:
            df["split"] = self.split
        return df
