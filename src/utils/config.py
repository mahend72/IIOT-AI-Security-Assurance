"""Config loading helpers.

All hyperparameters, dataset schema candidates, and mappings live in YAML
under configs/. Nothing here should hardcode a dataset-specific value —
that keeps the same pipeline code driving both ToN-IoT and Edge-IIoTset.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Load configs/<dataset_name>.yaml (e.g. 'toniot', 'edgeiiotset')."""
    path = CONFIGS_DIR / f"{dataset_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No config found for dataset '{dataset_name}' at {path}. "
            f"Expected one of: {[p.stem for p in CONFIGS_DIR.glob('*.yaml') if not p.stem.startswith('stage_mapping')]}"
        )
    return load_yaml(path)


def load_stage_mapping_config(dataset_name: str, variant: str = "primary") -> Dict[str, Any]:
    """variant: 'primary' (default, configs/stage_mapping_<dataset>.yaml) or a
    sensitivity variant name, e.g. 'conservative' / 'expanded' -> configs/
    stage_mapping_<dataset>_<variant>.yaml (see run_reviewer_experiments.py's
    stage_mapping_sensitivity experiment)."""
    suffix = "" if variant == "primary" else f"_{variant}"
    path = CONFIGS_DIR / f"stage_mapping_{dataset_name}{suffix}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No stage-mapping config found at {path} (dataset='{dataset_name}', variant='{variant}')"
        )
    return load_yaml(path)


def list_stage_mapping_variants(dataset_name: str) -> Dict[str, Path]:
    """{'primary': ..., 'conservative': ..., 'expanded': ...} for whichever
    variants actually exist on disk for this dataset."""
    variants = {"primary": CONFIGS_DIR / f"stage_mapping_{dataset_name}.yaml"}
    for p in CONFIGS_DIR.glob(f"stage_mapping_{dataset_name}_*.yaml"):
        variant_name = p.stem[len(f"stage_mapping_{dataset_name}_") :]
        variants[variant_name] = p
    return {k: v for k, v in variants.items() if v.exists()}


def override(base_cfg: Dict[str, Any], dotted_key: str, value: Any) -> Dict[str, Any]:
    """Return a deep-copied config with `dotted_key` (e.g. 'window.delta_t_seconds')
    set to `value`. Used by sensitivity-analysis scripts so we never mutate the
    loaded YAML in place."""
    cfg = copy.deepcopy(base_cfg)
    node = cfg
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = value
    return cfg
