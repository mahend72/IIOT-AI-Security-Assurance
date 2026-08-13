#!/usr/bin/env python3
"""Run the full stage-detection pipeline (RF + GCN + stacking meta-learner)
on one dataset and save metrics/figures to results/<dataset>/stage_detection/.

Example:
    python scripts/run_stage_detection.py --dataset toniot
    python scripts/run_stage_detection.py --dataset edgeiiotset --graph-mode temporal_only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.plotting import plot_confusion_matrix, plot_macro_f1_comparison
from src.evaluation.report import results_dir, save_csv, save_json
from src.evaluation.stage_metrics import compute_stage_metrics
from src.mapping.label_mapper import STAGE_ORDER
from src.pipeline import prepare_dataset
from src.training.stage_detector_trainer import train_stage_detector
from src.utils.logging_utils import get_logger
from src.utils.seed import set_global_seed
import pandas as pd
from sklearn.metrics import f1_score

logger = get_logger("run_stage_detection")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=["toniot", "edgeiiotset"])
    ap.add_argument("--graph-mode", default="both", choices=["interaction_only", "temporal_only", "both"])
    ap.add_argument("--meta-learner", default=None, choices=[None, "logistic_regression", "mlp", "gradient_boosting"])
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--delta-t", type=float, default=None, help="override window.delta_t_seconds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="", help="optional suffix for output filenames (used by sensitivity sweeps)")
    args = ap.parse_args()

    set_global_seed(args.seed)
    prepared = prepare_dataset(args.dataset, delta_t_seconds=args.delta_t, seed=None)

    result = train_stage_detector(
        prepared.windows_df, prepared.X, prepared.y, prepared.graph, prepared.cfg,
        graph_mode=args.graph_mode, n_folds=args.n_folds, seed=args.seed, meta_type=args.meta_learner,
    )

    out_dir = results_dir(args.dataset, "stage_detection")
    tag = f"_{args.tag}" if args.tag else ""

    all_metrics = {}
    macro_f1_test = {}
    for model_name, proba_dict in [("RF", result.rf_proba), ("GCN", result.gcn_proba), ("Stacked", result.stacked_proba)]:
        all_metrics[model_name] = {}
        for split in ("train", "val", "test"):
            y_true = result.y_true[split]
            y_pred = proba_dict[split].argmax(axis=1)
            m = compute_stage_metrics(y_true, y_pred, STAGE_ORDER)
            if split == "test":
                ci = bootstrap_ci(
                    lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
                    y_true, y_pred,
                    n_iterations=prepared.cfg["evaluation"]["bootstrap_iterations"],
                    ci=prepared.cfg["evaluation"]["bootstrap_ci"],
                    seed=args.seed,
                )
                m["macro_f1_bootstrap_ci"] = ci
                macro_f1_test[model_name] = m["macro_f1"]
            all_metrics[model_name][split] = m

    save_json(all_metrics, out_dir / f"stage_metrics{tag}.json")

    rows = []
    for model_name, splits in all_metrics.items():
        for split, m in splits.items():
            for cls, stats in m["per_class"].items():
                rows.append({"model": model_name, "split": split, "class": cls, **stats})
    save_csv(pd.DataFrame(rows), out_dir / f"stage_per_class_metrics{tag}.csv")

    summary_rows = [
        {"model": mn, "split": sp, "macro_f1": m["macro_f1"], "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"]}
        for mn, splits in all_metrics.items() for sp, m in splits.items()
    ]
    save_csv(pd.DataFrame(summary_rows), out_dir / f"stage_summary{tag}.csv")

    plot_confusion_matrix(
        all_metrics["Stacked"]["test"]["confusion_matrix"], STAGE_ORDER,
        f"{args.dataset}: Stacked model confusion matrix (test)", out_dir / f"confusion_matrix_stacked_test{tag}.png",
    )
    ci_bounds = {mn: (all_metrics[mn]["test"].get("macro_f1_bootstrap_ci", {}).get("lower", macro_f1_test[mn]),
                       all_metrics[mn]["test"].get("macro_f1_bootstrap_ci", {}).get("upper", macro_f1_test[mn]))
                 for mn in macro_f1_test}
    plot_macro_f1_comparison(
        macro_f1_test, f"{args.dataset}: Macro-F1 by model (test, {args.graph_mode})",
        out_dir / f"macro_f1_comparison{tag}.png", ci=ci_bounds,
    )

    logger.info(f"[{args.dataset}] Stage detection done. Test Macro-F1 -- RF: {macro_f1_test['RF']:.4f} | "
                f"GCN: {macro_f1_test['GCN']:.4f} | Stacked: {macro_f1_test['Stacked']:.4f}")
    logger.info(f"Results written to {out_dir}")
    return all_metrics


if __name__ == "__main__":
    main()
