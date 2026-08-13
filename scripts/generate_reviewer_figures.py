#!/usr/bin/env python3
"""Plots for the reviewer-response experiments (window / graph / meta-learner
sensitivity), read from the already-computed real-data CSVs in
results/<dataset>/reviewer_experiments/. No model is retrained here -- this
only visualizes numbers run_reviewer_experiments.py already produced.
Writes results/figures/<dataset>/{window,graph,meta_learner}_sensitivity.png
(skipped per-dataset if the underlying CSV is itself SKIPPED_WITH_REASON).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.evaluation.plotting import plot_macro_f1_comparison, plot_sensitivity_line
from src.utils.config import PROJECT_ROOT
from src.utils.logging_utils import get_logger

logger = get_logger("generate_reviewer_figures")


def main():
    for dataset in ["toniot", "edgeiiotset"]:
        rev_dir = PROJECT_ROOT / "results" / dataset / "reviewer_experiments"
        fig_dir = PROJECT_ROOT / "results" / "figures" / dataset

        window_csv = rev_dir / "window_sensitivity.csv"
        if window_csv.exists():
            df = pd.read_csv(window_csv)
            plot_sensitivity_line(
                x_values=df["delta_t_seconds"].tolist(),
                series={"Macro-F1": df["macro_f1"].tolist()},
                xlabel="Delta t (seconds)", ylabel="Macro-F1",
                title=f"{dataset}: window-size sensitivity",
                save_path=fig_dir / "window_sensitivity.png",
            )
            logger.info(f"[{dataset}] wrote window_sensitivity.png")
        else:
            logger.info(f"[{dataset}] window_sensitivity.csv absent (SKIPPED_WITH_REASON) -- no plot")

        graph_csv = rev_dir / "graph_ablation.csv"
        if graph_csv.exists():
            df = pd.read_csv(graph_csv)
            scores = dict(zip(df["graph_mode"], df["macro_f1"]))
            plot_macro_f1_comparison(scores, f"{dataset}: graph ablation", fig_dir / "graph_ablation.png")
            logger.info(f"[{dataset}] wrote graph_ablation.png")

        meta_csv = rev_dir / "meta_learner_ablation.csv"
        if meta_csv.exists():
            df = pd.read_csv(meta_csv)
            scores = dict(zip(df["meta_learner"], df["macro_f1"]))
            plot_macro_f1_comparison(scores, f"{dataset}: meta-learner ablation", fig_dir / "meta_learner_ablation.png")
            logger.info(f"[{dataset}] wrote meta_learner_ablation.png")


if __name__ == "__main__":
    main()
