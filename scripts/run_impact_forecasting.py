#!/usr/bin/env python3
"""Run the full impact-forecasting pipeline: stage detector (for pre-impact
evidence) -> dual-stream GRU forecaster, on one dataset. Saves metrics/
figures to results/<dataset>/impact_forecasting/.

For Edge-IIoTset, this script first checks the data-quality gate computed
by the adapter (src/data/edgeiiotset_adapter.py) — asset cardinality and
timestamp parseability — and if it fails, WRITES A REPORT explaining why
and exits without training a forecaster, per the project spec ("if
Edge-IIoTset lacks reliable fields for impact forecasting, still implement
stage detection and graph detection, then clearly report why full impact
forecasting is not valid").

Example:
    python scripts/run_impact_forecasting.py --dataset toniot
    python scripts/run_impact_forecasting.py --dataset edgeiiotset --horizon-multiple 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.forecast_metrics import compute_forecast_metrics
from src.evaluation.plotting import plot_pr_curves
from src.evaluation.report import results_dir, save_csv, save_json
from src.pipeline import prepare_dataset
from src.training.impact_forecast_trainer import build_row_stream_features, train_impact_forecaster
from src.training.sequence_builder import build_forecast_instances, instances_to_frame
from src.training.stage_detector_trainer import scatter_stage_proba, train_stage_detector
from src.utils.logging_utils import get_logger
from src.utils.seed import set_global_seed
from sklearn.metrics import average_precision_score, roc_auc_score

logger = get_logger("run_impact_forecasting")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=["toniot", "edgeiiotset"])
    ap.add_argument("--graph-mode", default="both", choices=["interaction_only", "temporal_only", "both"])
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--horizon-multiple", type=int, default=None, help="override forecasting.horizon_multiple")
    ap.add_argument("--delta-t", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true", help="run forecasting even if the data-quality gate fails")
    args = ap.parse_args()

    set_global_seed(args.seed)
    prepared = prepare_dataset(args.dataset, delta_t_seconds=args.delta_t)
    out_dir = results_dir(args.dataset, "impact_forecasting")
    tag = f"_{args.tag}" if args.tag else ""

    quality = prepared.bundle.quality
    if not args.force and quality.get("impact_forecasting_valid", True) is False:
        report = {
            "dataset": args.dataset,
            "impact_forecasting_run": False,
            "reasons": quality.get("impact_forecasting_invalid_reasons", []),
            "data_quality": quality,
            "note": ("Stage detection + graph construction ARE valid and were not affected. "
                     "Impact forecasting was skipped because the pre-impact-evidence sequences it "
                     "needs cannot be trusted for this dataset (see `reasons`). "
                     "Re-run with --force to attempt it anyway."),
        }
        save_json(report, out_dir / f"SKIPPED_report{tag}.json")
        logger.warning(f"[{args.dataset}] Impact forecasting SKIPPED: {report['reasons']}")
        logger.warning(f"Report written to {out_dir / f'SKIPPED_report{tag}.json'}")
        return report

    # Stage detector: supplies leakage-free (OOF/held-out) pre-impact evidence.
    stage_result = train_stage_detector(
        prepared.windows_df, prepared.X, prepared.y, prepared.graph, prepared.cfg,
        graph_mode=args.graph_mode, n_folds=args.n_folds, seed=args.seed,
    )
    stage_proba_full = scatter_stage_proba(prepared.windows_df, stage_result)

    fc_cfg = prepared.cfg["forecasting"]
    gru_cfg = prepared.cfg["models"]["gru_forecaster"]
    horizon_multiple = args.horizon_multiple or fc_cfg["horizon_multiple"]

    instances = build_forecast_instances(
        prepared.windows_df, prepared.split_map, horizon_multiple=horizon_multiple,
        max_seq_len=gru_cfg["max_seq_len"], drop_unconfirmed_negatives=fc_cfg["drop_unconfirmed_negatives"],
    )
    inst_summary = instances_to_frame(instances).groupby("split")["label"].agg(["count", "mean"]).reset_index()
    logger.info(f"[{args.dataset}] Forecast instances by split:\n{inst_summary.to_string(index=False)}")
    if inst_summary.empty or (inst_summary["count"] < 5).any():
        logger.warning(f"[{args.dataset}] Very few forecasting instances in some split(s) — "
                        f"metrics below may be unstable. Consider a smaller Delta t / horizon or more data.")

    feat_iad, feat_lmep = build_row_stream_features(prepared.X, stage_proba_full)
    forecast_result = train_impact_forecaster(instances, feat_iad, feat_lmep, prepared.cfg, seed=args.seed)

    capture_pcts = prepared.cfg["evaluation"]["capture_at_percents"]
    all_metrics = {}
    for split in ("train", "val", "test"):
        m = compute_forecast_metrics(forecast_result.y_true[split], forecast_result.y_proba[split], capture_pcts)
        if split == "test" and m["n_samples"] > 0:
            for metric_name, fn in [
                ("roc_auc", roc_auc_score), ("pr_auc", average_precision_score),
            ]:
                m[f"{metric_name}_bootstrap_ci"] = bootstrap_ci(
                    fn, forecast_result.y_true[split], forecast_result.y_proba[split],
                    n_iterations=prepared.cfg["evaluation"]["bootstrap_iterations"],
                    ci=prepared.cfg["evaluation"]["bootstrap_ci"], seed=args.seed,
                )
        all_metrics[split] = m

    all_metrics["instance_counts"] = inst_summary.to_dict("records")
    all_metrics["horizon_multiple"] = horizon_multiple
    all_metrics["delta_t_seconds"] = prepared.delta_t_seconds
    save_json(all_metrics, out_dir / f"forecast_metrics{tag}.json")

    save_csv(pd.DataFrame([{"split": s, **{k: v for k, v in all_metrics[s].items() if not isinstance(v, dict)}}
                           for s in ("train", "val", "test")]), out_dir / f"forecast_summary{tag}.csv")

    pred_rows = []
    for split in ("val", "test"):
        for a, w, yt, yp in zip(forecast_result.asset_ids[split], forecast_result.cut_window_ids[split],
                                 forecast_result.y_true[split], forecast_result.y_proba[split]):
            pred_rows.append({"split": split, "asset_id": a, "cut_window_id": w, "y_true": int(yt), "y_proba": float(yp)})
    save_csv(pd.DataFrame(pred_rows), out_dir / f"forecast_predictions{tag}.csv")

    plot_pr_curves(
        {"test": (forecast_result.y_true["test"], forecast_result.y_proba["test"])},
        f"{args.dataset}: Impact-forecast PR curve (H={horizon_multiple}xΔt, test)",
        out_dir / f"pr_curve_test{tag}.png",
    )

    logger.info(f"[{args.dataset}] Impact forecasting done. Test ROC-AUC={all_metrics['test'].get('roc_auc'):.4f} "
                f"PR-AUC={all_metrics['test'].get('pr_auc'):.4f} F1={all_metrics['test'].get('f1'):.4f}")
    logger.info(f"Results written to {out_dir}")
    return all_metrics


if __name__ == "__main__":
    main()
