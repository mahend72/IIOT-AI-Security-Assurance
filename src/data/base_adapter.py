"""Shared logic for dataset adapters: schema inspection + column matching.

Design rule (per project spec): "Do not invent dataset columns." Every
adapter MUST inspect the real CSV header of the file(s) it is given and
match against config-provided *candidate* names — it never assumes a
candidate is present. Required-but-missing fields raise a clear,
actionable error; optional-but-missing fields degrade gracefully and are
recorded in `DatasetBundle.warnings` / `.quality`.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from src.utils.config import PROJECT_ROOT
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _normalize(s: str) -> str:
    """Lowercase and strip everything except alphanumerics so that
    'ip.src_host', 'ip_src_host', and 'IP Src Host' all compare equal."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def match_column(actual_columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """Return the first actual column name matching any candidate, or None."""
    norm_to_actual: Dict[str, str] = {_normalize(c): c for c in actual_columns}
    for cand in candidates:
        norm_cand = _normalize(cand)
        if norm_cand in norm_to_actual:
            return norm_to_actual[norm_cand]
    return None


def match_columns(actual_columns: Iterable[str], candidate_lists: Dict[str, List[str]]) -> Dict[str, Optional[str]]:
    """Vectorized match_column over several {field_name: [candidates]}."""
    return {field: match_column(actual_columns, cands) for field, cands in candidate_lists.items()}


def resolve_raw_files(raw_dir: str, preferred_filename: str, file_glob: str) -> List[Path]:
    """Find the raw CSV file(s) for a dataset: prefer the documented exact
    filename, else fall back to globbing the raw_dir for any CSVs."""
    base = PROJECT_ROOT / raw_dir
    preferred = base / preferred_filename
    if preferred.exists():
        return [preferred]
    found = sorted(Path(p) for p in glob.glob(str(base / file_glob)))
    if not found:
        raise FileNotFoundError(
            f"No raw data files found in {base} (looked for '{preferred_filename}' "
            f"and glob '{file_glob}'). Download the dataset and place the CSV(s) "
            f"there — see the dataset's `source_note` in its config."
        )
    return found


def load_concat_csvs(files: List[Path], **read_csv_kwargs) -> pd.DataFrame:
    frames = []
    for f in files:
        logger.info(f"Reading {f} ...")
        frames.append(pd.read_csv(f, low_memory=False, **read_csv_kwargs))
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} raw columns from {len(files)} file(s).")
    return df
