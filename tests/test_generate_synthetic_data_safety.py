"""Regression tests for the overwrite-safety guard in
scripts/generate_synthetic_data.py.

Added after an incident where running the generator against a directory
that already held real ToN-IoT/Edge-IIoTset CSVs silently overwrote them
with synthetic data. These tests pin the fix: the generator must never
overwrite a file that is not itself its own prior output, unless the
caller explicitly passes --force (and even then, it must warn loudly).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_SCRIPT = PROJECT_ROOT / "scripts" / "generate_synthetic_data.py"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import generate_synthetic_data as gen  # noqa: E402


def _run_generator(out_root: Path, *extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, str(GENERATOR_SCRIPT),
            "--out-root", str(out_root),
            "--n-assets", "4", "--hours", "1.0", "--rate-hz", "0.05", "--seed", "1",
            *extra_args,
        ],
        capture_output=True, text=True,
    )


# ---------------------------------------------------------------------------
# Unit-level: the guard function itself
# ---------------------------------------------------------------------------

def test_guard_allows_write_when_target_does_not_exist(tmp_path):
    target = tmp_path / "toniot" / "Train_Test_Network.csv"
    gen._refuse_if_unsafe_overwrite(target, force=False)  # must not raise


def test_guard_refuses_real_looking_file_without_force(tmp_path):
    d = tmp_path / "toniot"
    d.mkdir()
    real_csv = d / "Train_Test_Network.csv"
    real_csv.write_text("ts,src_ip,label,type\n1700000000,10.0.0.1,0,normal\n")

    with pytest.raises(gen.RealDataOverwriteError, match="Refusing to overwrite"):
        gen._refuse_if_unsafe_overwrite(real_csv, force=False)

    # the guard itself must never touch the file it is protecting
    assert "1700000000" in real_csv.read_text()


def test_guard_allows_regenerating_its_own_prior_output_without_force(tmp_path):
    d = tmp_path / "toniot"
    d.mkdir()
    csv_path = d / "Train_Test_Network.csv"
    csv_path.write_text("previous synthetic run\n")
    (d / gen.SYNTHETIC_MARKER_NAME).write_text("{}")

    gen._refuse_if_unsafe_overwrite(csv_path, force=False)  # must not raise


def test_guard_permits_overwrite_with_force_and_warns(tmp_path, capsys):
    d = tmp_path / "toniot"
    d.mkdir()
    real_csv = d / "Train_Test_Network.csv"
    real_csv.write_text("possibly real data\n")

    gen._refuse_if_unsafe_overwrite(real_csv, force=True)  # must not raise

    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "may contain a real dataset" in captured.err


# ---------------------------------------------------------------------------
# CLI-level: the actual `python scripts/generate_synthetic_data.py` entry point
# ---------------------------------------------------------------------------

def test_cli_refuses_to_overwrite_existing_real_looking_dataset(tmp_path):
    real_dir = tmp_path / "toniot"
    real_dir.mkdir()
    real_csv = real_dir / "Train_Test_Network.csv"
    original_content = "ts,src_ip,dst_ip,label,type\n1700000000,10.0.0.1,10.0.0.2,0,normal\n"
    real_csv.write_text(original_content)

    result = _run_generator(tmp_path)

    assert result.returncode != 0
    assert "Refusing to overwrite" in result.stderr
    # the "real" file and the untouched edgeiiotset dir must both be intact
    assert real_csv.read_text() == original_content
    assert not (tmp_path / "edgeiiotset" / "ML-EdgeIIoT-dataset.csv").exists()


def test_cli_writes_normally_into_an_empty_dedicated_directory(tmp_path):
    result = _run_generator(tmp_path)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "toniot" / "Train_Test_Network.csv").exists()
    assert (tmp_path / "toniot" / gen.SYNTHETIC_MARKER_NAME).exists()
    assert (tmp_path / "edgeiiotset" / "ML-EdgeIIoT-dataset.csv").exists()


def test_cli_can_regenerate_its_own_prior_output_without_force(tmp_path):
    first = _run_generator(tmp_path)
    assert first.returncode == 0, first.stderr

    second = _run_generator(tmp_path)  # same target dir, now marked synthetic
    assert second.returncode == 0, second.stderr


def test_cli_overwrites_real_looking_data_only_with_force_and_warns(tmp_path):
    real_dir = tmp_path / "toniot"
    real_dir.mkdir()
    (real_dir / "Train_Test_Network.csv").write_text("possibly real, no marker\n")

    result = _run_generator(tmp_path, "--force")

    assert result.returncode == 0, result.stderr
    assert "WARNING" in result.stderr
    assert (tmp_path / "toniot" / gen.SYNTHETIC_MARKER_NAME).exists()


def test_cli_default_out_root_is_data_raw_unchanged():
    """The default --out-root must remain data/raw -- the documented local
    smoke-test workflow (README's Datasets section) is unchanged."""
    result = subprocess.run(
        [sys.executable, str(GENERATOR_SCRIPT), "--help"], capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "data/raw" in result.stdout
