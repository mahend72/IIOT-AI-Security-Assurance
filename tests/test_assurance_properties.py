"""CI-facing assurance-property tests.

These exercise the project's core leakage/assurance guarantees against a
tiny SYNTHETIC dataset built entirely inside pytest's `tmp_path` (via
`scripts/generate_synthetic_data.py`'s generator functions) -- never
touching `data/raw/<dataset>/`. That keeps them safe to run in a fresh
checkout with no real ToN-IoT/Edge-IIoTset data (as in CI) and equally safe
to run on a machine that already has the real datasets in place locally.

Most checks are imported directly from `scripts/run_final_validation.py`
(the project's own runtime assurance auditor) rather than reimplemented, so
this file and that script can never silently drift apart.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import generate_synthetic_data as gen  # noqa: E402
import run_final_validation as rfv  # noqa: E402

import src.utils.data_provenance as data_provenance  # noqa: E402
from src.data.loader import load_dataset  # noqa: E402
from src.mapping.label_mapper import STAGE_ORDER  # noqa: E402
from src.pipeline import prepare_dataset  # noqa: E402
from src.preprocessing.splitting import asset_disjoint_split  # noqa: E402
from src.utils.config import list_stage_mapping_variants, load_dataset_config, load_yaml, override  # noqa: E402


def _build_synthetic_prepared(dataset_name: str, tmp_path: Path, seed: int = 7):
    """Build a real PreparedDataset (src/pipeline.py) from a small synthetic
    CSV written under `tmp_path`, reusing the actual generator/adapter code
    paths but pointed away from data/raw/<dataset_name>/ via a config
    override -- so this can never read or overwrite real project data."""
    raw_dir = tmp_path / f"raw_{dataset_name}"
    raw_dir.mkdir(parents=True)
    if dataset_name == "toniot":
        gen.generate_toniot(
            raw_dir / "Train_Test_Network.csv",
            n_assets=20, attacked_frac=0.4, seed=seed, hours=3.0, rate_hz=0.05,
        )
    else:
        gen.generate_edgeiiotset(
            raw_dir / "ML-EdgeIIoT-dataset.csv",
            n_assets=20, attacked_frac=0.4, seed=seed, hours=3.0, rate_hz=0.05,
        )

    load_cfg = override(load_dataset_config(dataset_name), "dataset.raw_dir", str(raw_dir))
    bundle = load_dataset(dataset_name, load_cfg)
    return prepare_dataset(dataset_name, seed=seed, preloaded_bundle=bundle)


@pytest.fixture(scope="module")
def synthetic_toniot(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("assurance_toniot")
    return _build_synthetic_prepared("toniot", tmp_path)


# ---------------------------------------------------------------------------
# Asset-disjoint splitting / no identity leakage across partitions
# ---------------------------------------------------------------------------

def test_asset_disjoint_split_assigns_each_asset_exactly_once():
    asset_ids = [f"asset-{i}" for i in range(50) for _ in range(3)]  # 3 records/asset
    split_map = asset_disjoint_split(asset_ids, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=1)
    assert set(split_map) == {f"asset-{i}" for i in range(50)}
    assert set(split_map.values()) <= {"train", "val", "test"}


def test_no_identity_leakage_across_partitions(synthetic_toniot):
    result = rfv.check_no_asset_in_multiple_splits(synthetic_toniot)
    assert result["pass"], result["detail"]


# ---------------------------------------------------------------------------
# Preprocessing fitted only on training data
# ---------------------------------------------------------------------------

def test_preprocessing_fitted_only_on_train(synthetic_toniot):
    result = rfv.check_scaler_fit_train_only(synthetic_toniot)
    assert result["pass"], result["detail"]


# ---------------------------------------------------------------------------
# No IMP-stage evidence entering forecasting inputs
# ---------------------------------------------------------------------------

def test_no_imp_evidence_in_forecast_prefixes(synthetic_toniot):
    result = rfv.check_forecaster_pre_cutoff_only(synthetic_toniot)
    assert result["pass"], result["detail"]


def test_no_imp_evidence_in_forecast_features(synthetic_toniot):
    result = rfv.check_no_imp_evidence_in_forecaster_features(synthetic_toniot)
    assert result["pass"], result["detail"]


# ---------------------------------------------------------------------------
# No cross-split graph edges / neighbourhood leakage
# ---------------------------------------------------------------------------

def test_oof_subgraph_is_train_only(synthetic_toniot):
    result = rfv.check_oof_subgraph_train_only(synthetic_toniot)
    assert result["pass"], result["detail"]


def test_full_graph_can_contain_cross_split_edges_but_induced_subgraph_cannot(synthetic_toniot):
    """Confirms the check above is non-trivial: the FULL asset-time graph
    does connect nodes across different splits (assets in different splits
    can still have communicated with each other), so the train-only
    induced_subgraph property is actually guarding against something, not
    passing vacuously on an already split-disjoint graph."""
    graph = synthetic_toniot.graph
    split = graph.node_table["split"].to_numpy()
    edges = graph.edge_index_interaction.numpy()
    if edges.shape[1] == 0:
        pytest.skip("synthetic sample produced no interaction edges to check")
    crosses_split = bool((split[edges[0]] != split[edges[1]]).any())
    assert crosses_split, "expected >=1 cross-split edge in the full graph for this guard to be meaningful"


# ---------------------------------------------------------------------------
# Valid stage-mapping configuration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset_name", ["toniot", "edgeiiotset"])
def test_stage_mapping_configs_are_valid(dataset_name):
    variants = list_stage_mapping_variants(dataset_name)
    assert variants, f"no stage_mapping config found for {dataset_name}"
    valid_stages = set(STAGE_ORDER) - {"Benign"}
    for path in variants.values():
        cfg = load_yaml(path)
        assert cfg.get("benign_values"), f"{path}: benign_values must be non-empty"
        mapping = cfg.get("mapping", {})
        assert mapping, f"{path}: mapping must be non-empty"
        for raw_value, stage in mapping.items():
            assert stage in valid_stages, f"{path}: mapping['{raw_value}']='{stage}' not in {sorted(valid_stages)}"
        default_unmapped = cfg.get("default_unmapped")
        assert default_unmapped in valid_stages, f"{path}: default_unmapped='{default_unmapped}' not in {sorted(valid_stages)}"


# ---------------------------------------------------------------------------
# Synthetic data clearly separated from scientific real-data outputs
# ---------------------------------------------------------------------------

def test_synthetic_marker_written_by_generator(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    args = argparse.Namespace(n_assets=4, attacked_frac=0.4, hours=1.0, rate_hz=0.05, seed=1, clean_markers=False)
    gen._write_synthetic_marker(raw_dir, args)
    assert (raw_dir / ".SYNTHETIC_DATA_MARKER").exists()


def test_require_real_data_refuses_synthetic_marked_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(data_provenance, "PROJECT_ROOT", tmp_path)
    (tmp_path / "data" / "raw" / "fakeds").mkdir(parents=True)

    assert not data_provenance.is_synthetic_data("fakeds")
    data_provenance.synthetic_marker_path("fakeds").write_text("{}")
    assert data_provenance.is_synthetic_data("fakeds")

    with pytest.raises(data_provenance.SyntheticDataGuardError):
        data_provenance.require_real_data("fakeds")
    data_provenance.require_real_data("fakeds", allow_synthetic=True)  # must not raise
