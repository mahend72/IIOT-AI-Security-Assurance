"""Guards against ever writing synthetic smoke-test data into a manuscript
results path.

`scripts/generate_synthetic_data.py` drops a `.SYNTHETIC_DATA_MARKER` file
next to whatever CSV(s) it writes in `data/raw/<dataset>/`. Every
manuscript-facing orchestration script (run_main_results.py,
run_reviewer_experiments.py) calls `require_real_data` before writing
anything under results/<dataset>/main/, results/<dataset>/reviewer_experiments/,
or results/manuscript_tables/ — if the marker is present, it refuses to
proceed unless the caller explicitly passes --allow-synthetic (only ever
used by the code-verification pass, which writes to a separate
--output-root and never touches the real manuscript paths).
"""
from __future__ import annotations

from pathlib import Path

from src.utils.config import PROJECT_ROOT

SYNTHETIC_MARKER_NAME = ".SYNTHETIC_DATA_MARKER"


def synthetic_marker_path(dataset_name: str) -> Path:
    return PROJECT_ROOT / "data" / "raw" / dataset_name / SYNTHETIC_MARKER_NAME


def is_synthetic_data(dataset_name: str) -> bool:
    return synthetic_marker_path(dataset_name).exists()


class SyntheticDataGuardError(RuntimeError):
    pass


def require_real_data(dataset_name: str, allow_synthetic: bool = False) -> None:
    """Raise SyntheticDataGuardError if `dataset_name`'s raw data directory
    still contains the synthetic-generator marker, unless the caller
    explicitly opted into `allow_synthetic` (code-verification runs only)."""
    if is_synthetic_data(dataset_name) and not allow_synthetic:
        raise SyntheticDataGuardError(
            f"data/raw/{dataset_name}/ still contains {SYNTHETIC_MARKER_NAME} — this is "
            f"scripts/generate_synthetic_data.py smoke-test data, not the real dataset. "
            f"Refusing to write manuscript results from it. Replace the CSV(s) with the "
            f"real dataset (this also deletes the marker automatically if you copy a real "
            f"file over it -- or run scripts/generate_synthetic_data.py --clean-markers "
            f"after placing real data), or pass --allow-synthetic if this is intentionally "
            f"a code-verification run writing to a scratch --output-root."
        )
