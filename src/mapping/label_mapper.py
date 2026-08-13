"""Map raw per-record attack-type strings to the paper's 4-class stage
taxonomy: Benign, IAD, LMEP, IMP.

The mapping tables themselves live in configs/stage_mapping_<dataset>.yaml
(see those files for the documented rationale) — this module only applies
them, with a normalization pass so that minor formatting differences
('DDoS_UDP' vs 'ddos udp' vs 'DDOS-UDP') don't cause silent misses.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

import pandas as pd

STAGE_ORDER = ["Benign", "IAD", "LMEP", "IMP"]
STAGE_SEVERITY = {s: i for i, s in enumerate(STAGE_ORDER)}


def _normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


class LabelMapper:
    def __init__(self, stage_mapping_config: Dict[str, Any]):
        self.benign_values = {_normalize_key(v) for v in stage_mapping_config.get("benign_values", [])}
        self.mapping = {_normalize_key(k): v for k, v in stage_mapping_config.get("mapping", {}).items()}
        self.default_unmapped = stage_mapping_config.get("default_unmapped", "IAD")
        self._unmapped_seen: set[str] = set()

    def map_value(self, raw_value: str) -> str:
        key = _normalize_key(raw_value)
        if key in self.benign_values:
            return "Benign"
        if key in self.mapping:
            return self.mapping[key]
        self._unmapped_seen.add(str(raw_value))
        return self.default_unmapped

    def map_series(self, raw_series: pd.Series) -> pd.Series:
        return raw_series.map(self.map_value).astype("category")

    @property
    def unmapped_values_seen(self) -> List[str]:
        return sorted(self._unmapped_seen)


def stage_to_severity(stage: str) -> int:
    return STAGE_SEVERITY[stage]


def max_severity_stage(stages: Iterable[str]) -> str:
    """Aggregate several per-record stage labels into one, taking the worst
    (highest-severity) stage present — used for window-level labeling."""
    best = "Benign"
    best_sev = 0
    for s in stages:
        sev = STAGE_SEVERITY.get(s, 0)
        if sev > best_sev:
            best_sev = sev
            best = s
    return best
