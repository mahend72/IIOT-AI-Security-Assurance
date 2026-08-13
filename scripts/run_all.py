#!/usr/bin/env python3
"""Run the complete pipeline (stage detection + impact forecasting +
reviewer-response sensitivity experiments) for both ToN-IoT and
Edge-IIoTset, in one command.

Each stage is invoked as a SEPARATE process (same way you'd run it by
hand) so a failure/skip in one stage never corrupts state for another, and
so this script's behavior always matches the individual commands documented
in the README.

Example:
    python scripts/run_all.py                      # full run, both datasets
    python scripts/run_all.py --quick               # fewer folds/epochs, for smoke-testing
    python scripts/run_all.py --datasets toniot      # just one dataset
    python scripts/run_all.py --skip-sensitivity     # skip the (slow) reviewer-response sweeps
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger("run_all")


def run(cmd: list[str]) -> int:
    logger.info(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        logger.error(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["toniot", "edgeiiotset"], choices=["toniot", "edgeiiotset"])
    ap.add_argument("--quick", action="store_true", help="fewer CV folds, for a fast smoke-test run")
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n_folds = "3" if args.quick else "5"
    py = sys.executable
    failures = []

    for dataset in args.datasets:
        logger.info(f"===== {dataset}: stage detection =====")
        rc = run([py, "scripts/run_stage_detection.py", "--dataset", dataset, "--n-folds", n_folds, "--seed", str(args.seed)])
        if rc != 0:
            failures.append(f"{dataset}/stage_detection")

        logger.info(f"===== {dataset}: impact forecasting =====")
        rc = run([py, "scripts/run_impact_forecasting.py", "--dataset", dataset, "--n-folds", n_folds, "--seed", str(args.seed)])
        if rc != 0:
            failures.append(f"{dataset}/impact_forecasting")

        if not args.skip_sensitivity:
            logger.info(f"===== {dataset}: sensitivity / reviewer-response experiments =====")
            sens_cmd = [py, "scripts/run_sensitivity.py", "--dataset", dataset, "--n-folds", n_folds, "--seed", str(args.seed)]
            if args.quick:
                sens_cmd += ["--skip-tuning"]
            rc = run(sens_cmd)
            if rc != 0:
                failures.append(f"{dataset}/sensitivity")

    if failures:
        logger.error(f"Completed with failures in: {failures}")
        sys.exit(1)
    logger.info("All requested stages completed successfully. See results/<dataset>/ for metrics, CSVs, and figures.")


if __name__ == "__main__":
    main()
