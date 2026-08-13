"""Small helpers for writing metrics to results/ as CSV/JSON — kept in one
place so every script saves in the same shape and directory convention."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import PROJECT_ROOT


def results_dir(dataset_name: str, sub: str = "") -> Path:
    d = PROJECT_ROOT / "results" / dataset_name / sub if sub else PROJECT_ROOT / "results" / dataset_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _json_default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
