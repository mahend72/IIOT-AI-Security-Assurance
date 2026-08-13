#!/usr/bin/env python3
"""Manuscript MAIN results for one dataset: dataset summary, stage-detection
model comparison (RF / GCN / late fusion / stacked RF-GCN / XGBoost /
LightGBM / GraphSAGE / GAT / no-graph GRU temporal baseline), one-vs-rest
stage alerting (IAD/LMEP/IMP vs rest), and impact forecasting (or a
SKIPPED_WITH_REASON report if not valid for this dataset).

Writes:
  results/<dataset>/main/dataset_summary.csv
  results/<dataset>/main/stage_detection_main.csv
  results/<dataset>/main/one_vs_rest_alerting.csv
  results/<dataset>/main/impact_forecasting.csv                         (if valid)
  results/<dataset>/main/impact_forecasting_SKIPPED_WITH_REASON.json    (if not)
  results/manuscript_tables/tab_<dataset>_dataset_summary.tex
  results/manuscript_tables/tab_<dataset>_stage_detection.tex
  results/manuscript_tables/tab_<dataset>_ovr_alerting.tex
  results/manuscript_tables/tab_<dataset>_impact_forecasting.tex
  results/figures/<dataset>/*.png

Refuses to run (SyntheticDataGuardError) if data/raw/<dataset>/ still has
the synthetic-generator marker -- see src/utils/data_provenance.py.

Example:
    python scripts/run_main_results.py --dataset toniot
    python scripts/run_main_results.py --dataset edgeiiotset --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.dataset_summary import dataset_summary_dataframe
from src.evaluation.forecast_metrics import compute_forecast_metrics
from src.evaluation.forecast_validity import check_instance_level_validity, instance_summary
from src.evaluation.latex_tables import df_to_latex_table, write_skipped_latex_note
from src.evaluation.ovr_metrics import one_vs_rest_alerting_metrics
from src.evaluation.plotting import (
    plot_confusion_matrix,
    plot_macro_f1_comparison,
    plot_pr_curves,
    plot_sensitivity_line,
)
from src.evaluation.stage_metrics import compute_stage_metrics
from src.mapping.label_mapper import STAGE_ORDER
from src.pipeline import prepare_dataset
from src.training.impact_forecast_trainer import build_row_stream_features, train_impact_forecaster
from src.training.model_comparison import run_all_stage_models
from src.training.sequence_builder import build_forecast_instances
from src.training.stage_detector_trainer import scatter_stage_proba, train_stage_detector
from src.utils.data_provenance import require_real_data
from src.utils.logging_utils import get_logger
from src.utils.run_metadata import RunMetadata
from src.utils.seed import set_global_seed

logger = get_logger("run_main_results")

MODEL_DISPLAY_NAMES = {
    "RF": "RF (feature-only)",
    "GCN": "GCN (graph-only)",
    "Late_Fusion_RF_GCN": "Late fusion (RF+GCN)",
    "Stacked_RF_GCN": "Stacked RF-GCN",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "GRU_Temporal_NoGraph": "GRU (no-graph temporal)",
    "GraphSAGE": "GraphSAGE",
    "GAT": "GAT",
}
MAIN_STAGE_MODEL_ORDER = ["RF", "GCN", "Late_Fusion_RF_GCN", "Stacked_RF_GCN", "XGBoost", "LightGBM",
                          "GRU_Temporal_NoGraph", "GraphSAGE", "GAT"]


def results_root(dataset: str) -> Path:
    from src.utils.config import PROJECT_ROOT
    return PROJECT_ROOT / "results"


def section_dataset_summary(dataset: str, prepared, meta: RunMetadata, out_main: Path, out_tables: Path):
    df = dataset_summary_dataframe(prepared)
    df_stamped = meta.stamp_df(df)
    df_stamped.to_csv(out_main / "dataset_summary.csv", index=False)
    df_to_latex_table(
        df, caption=f"{dataset}: dataset summary.", label=f"tab:{dataset}_dataset_summary",
        save_path=out_tables / f"tab_{dataset}_dataset_summary.tex",
        columns=["metric", "value", "note"], column_headers=["Metric", "Value", "Note"],
    )
    logger.info(f"[{dataset}] dataset summary written.")
    return df


def section_stage_detection(dataset: str, prepared, seeds, n_folds, meta: RunMetadata, out_main: Path, out_tables: Path, out_figs: Path):
    logger.info(f"[{dataset}] Training {len(MAIN_STAGE_MODEL_ORDER)} models x {len(seeds)} seeds for stage detection...")
    results = run_all_stage_models(prepared, seeds=seeds, n_folds=n_folds, include_temporal_baseline=True)

    rows = []
    per_class_rows = []
    test_macro_f1_seed0 = {}
    for model_name in MAIN_STAGE_MODEL_ORDER:
        if model_name not in results:
            rows.append({"model": MODEL_DISPLAY_NAMES[model_name], "macro_f1_mean": np.nan, "macro_f1_std": np.nan,
                         "note": "N/A for this dataset (see data-quality note)."})
            continue
        macro_f1s, f1_by_class = [], {c: [] for c in STAGE_ORDER}
        for seed, splits in results[model_name].items():
            te = splits["test"]
            if len(te["y_true"]) == 0:
                continue
            y_pred = te["proba"].argmax(axis=1)
            m = compute_stage_metrics(te["y_true"], y_pred, STAGE_ORDER)
            macro_f1s.append(m["macro_f1"])
            for c in STAGE_ORDER:
                f1_by_class[c].append(m["per_class"][c]["f1"])
            if seed == seeds[0]:
                test_macro_f1_seed0[model_name] = m["macro_f1"]

        row = {
            "model": MODEL_DISPLAY_NAMES[model_name],
            "macro_f1_mean": float(np.mean(macro_f1s)) if macro_f1s else np.nan,
            "macro_f1_std": float(np.std(macro_f1s)) if len(macro_f1s) > 1 else 0.0,
            "n_seeds": len(macro_f1s),
        }
        for c in STAGE_ORDER:
            row[f"f1_{c.lower()}_mean"] = float(np.mean(f1_by_class[c])) if f1_by_class[c] else np.nan
            row[f"f1_{c.lower()}_std"] = float(np.std(f1_by_class[c])) if len(f1_by_class[c]) > 1 else 0.0
        rows.append(row)
        for c in STAGE_ORDER:
            per_class_rows.append({"model": MODEL_DISPLAY_NAMES[model_name], "class": c,
                                   "f1_mean": row[f"f1_{c.lower()}_mean"], "f1_std": row[f"f1_{c.lower()}_std"]})

    df = pd.DataFrame(rows)
    meta.stamp_df(df).to_csv(out_main / "stage_detection_main.csv", index=False)
    df_to_latex_table(
        df, caption=f"{dataset}: four-class stage detection (test, mean$\\pm$std over {len(seeds)} seeds).",
        label=f"tab:{dataset}_stage_detection", save_path=out_tables / f"tab_{dataset}_stage_detection.tex",
        columns=["model", "macro_f1_mean", "macro_f1_std", "f1_benign_mean", "f1_iad_mean", "f1_lmep_mean", "f1_imp_mean", "n_seeds"],
        column_headers=["Model", "Macro-F1", "Std", "F1(Benign)", "F1(IAD)", "F1(LMEP)", "F1(IMP)", "N seeds"],
    )

    # Figures: confusion matrix (Stacked, seed[0]) + Macro-F1 ranking.
    if "Stacked_RF_GCN" in results and seeds[0] in results["Stacked_RF_GCN"]:
        te = results["Stacked_RF_GCN"][seeds[0]]["test"]
        if len(te["y_true"]):
            cm_metrics = compute_stage_metrics(te["y_true"], te["proba"].argmax(axis=1), STAGE_ORDER)
            plot_confusion_matrix(cm_metrics["confusion_matrix"], STAGE_ORDER,
                                   f"{dataset}: Stacked RF-GCN confusion matrix (test)",
                                   out_figs / "confusion_matrix_stacked_test.png")
    ranking = {MODEL_DISPLAY_NAMES[m]: test_macro_f1_seed0[m] for m in MAIN_STAGE_MODEL_ORDER if m in test_macro_f1_seed0}
    if ranking:
        plot_macro_f1_comparison(ranking, f"{dataset}: Macro-F1 by model (test)", out_figs / "macro_f1_model_ranking.png")

    return df, results


def section_one_vs_rest(dataset: str, results, meta: RunMetadata, out_main: Path, out_tables: Path, out_figs: Path, seeds):
    stages = ["IAD", "LMEP", "IMP"]
    if "Stacked_RF_GCN" not in results or seeds[0] not in results["Stacked_RF_GCN"]:
        logger.warning(f"[{dataset}] No Stacked_RF_GCN result available for one-vs-rest alerting.")
        return pd.DataFrame()
    splits = results["Stacked_RF_GCN"][seeds[0]]
    y_true = {s: splits[s]["y_true"] for s in ("train", "val", "test")}
    proba = {s: splits[s]["proba"] for s in ("train", "val", "test")}
    rows = one_vs_rest_alerting_metrics(y_true, proba, STAGE_ORDER, stages)
    df = pd.DataFrame(rows)
    meta.stamp_df(df).to_csv(out_main / "one_vs_rest_alerting.csv", index=False)
    df_to_latex_table(
        df[df["split"] == "test"], caption=f"{dataset}: one-vs-rest stage alerting (Stacked RF-GCN, test).",
        label=f"tab:{dataset}_ovr_alerting", save_path=out_tables / f"tab_{dataset}_ovr_alerting.tex",
        columns=["stage", "precision", "recall", "f1", "pr_auc", "threshold", "n_positive", "n_total"],
        column_headers=["Stage", "Precision", "Recall", "F1", "PR-AUC", "Threshold", "N pos", "N total"],
        note=rows[0]["threshold_selection_rule"] if rows else None,
    )
    curves = {}
    for stage in stages:
        idx = STAGE_ORDER.index(stage)
        yt = (y_true["test"] == idx).astype(int)
        ys = proba["test"][:, idx]
        curves[stage] = (yt, ys)
    plot_pr_curves(curves, f"{dataset}: one-vs-rest PR curves (test)", out_figs / "ovr_pr_curves_test.png")
    return df


def section_impact_forecasting(dataset: str, prepared, seeds, n_folds, meta: RunMetadata, out_main: Path, out_tables: Path, out_figs: Path):
    q = prepared.bundle.quality
    if not q.get("impact_forecasting_valid", True):
        reasons = q.get("impact_forecasting_invalid_reasons", [])
        report = meta.stamp_json({
            "dataset": dataset, "experiment": "impact_forecasting", "run": False,
            "gate": "metadata (asset cardinality / timestamp reliability)", "reasons": reasons,
        })
        import json
        (out_main / "impact_forecasting_SKIPPED_WITH_REASON.json").write_text(json.dumps(report, indent=2))
        write_skipped_latex_note(
            f"{dataset}: impact forecasting -- SKIPPED.", f"tab:{dataset}_impact_forecasting",
            out_tables / f"tab_{dataset}_impact_forecasting.tex", reasons, dataset,
        )
        logger.warning(f"[{dataset}] Impact forecasting SKIPPED (metadata gate): {reasons}")
        return None

    fc_cfg = prepared.cfg["forecasting"]
    gru_cfg = prepared.cfg["models"]["gru_forecaster"]
    instances = build_forecast_instances(
        prepared.windows_df, prepared.split_map, horizon_multiple=fc_cfg["horizon_multiple"],
        max_seq_len=gru_cfg["max_seq_len"], drop_unconfirmed_negatives=fc_cfg["drop_unconfirmed_negatives"],
    )
    ok, reasons = check_instance_level_validity(instances)
    if not ok:
        report = meta.stamp_json({
            "dataset": dataset, "experiment": "impact_forecasting", "run": False,
            "gate": "instance-level (post sequence-construction)", "reasons": reasons,
            "instance_summary": instance_summary(instances),
        })
        import json
        (out_main / "impact_forecasting_SKIPPED_WITH_REASON.json").write_text(json.dumps(report, indent=2))
        write_skipped_latex_note(
            f"{dataset}: impact forecasting -- SKIPPED.", f"tab:{dataset}_impact_forecasting",
            out_tables / f"tab_{dataset}_impact_forecasting.tex", reasons, dataset,
        )
        logger.warning(f"[{dataset}] Impact forecasting SKIPPED (instance-level gate): {reasons}")
        return None

    capture_pcts = prepared.cfg["evaluation"]["capture_at_percents"]
    rows = []
    curves = {}
    for seed in seeds:
        set_global_seed(seed)
        stage_result = train_stage_detector(prepared.windows_df, prepared.X, prepared.y, prepared.graph, prepared.cfg, n_folds=n_folds, seed=seed)
        stage_proba_full = scatter_stage_proba(prepared.windows_df, stage_result)
        feat_iad, feat_lmep = build_row_stream_features(prepared.X, stage_proba_full)

        seed_instances = build_forecast_instances(
            prepared.windows_df, prepared.split_map, horizon_multiple=fc_cfg["horizon_multiple"],
            max_seq_len=gru_cfg["max_seq_len"], drop_unconfirmed_negatives=fc_cfg["drop_unconfirmed_negatives"],
        )
        fres = train_impact_forecaster(seed_instances, feat_iad, feat_lmep, prepared.cfg, seed=seed)

        for model_name, y_proba_fn in [
            ("Dual-stream GRU (main)", lambda: fres.y_proba["test"]),
        ]:
            m = compute_forecast_metrics(fres.y_true["test"], y_proba_fn(), capture_pcts)
            m["model"] = model_name
            m["seed"] = seed
            m["n_positive"] = int(fres.y_true["test"].sum())
            m["n_negative"] = int((fres.y_true["test"] == 0).sum())
            rows.append(m)
        if seed == seeds[0]:
            curves["Dual-stream GRU (test)"] = (fres.y_true["test"], fres.y_proba["test"])

    # Simple ordinal baselines: Max-Stage2 / Count-Stage2 (using LMEP evidence
    # only, no learned forecaster) -- computed once (seed-independent, since
    # they are non-learned).
    lmep_idx = STAGE_ORDER.index("LMEP")
    for seed in [seeds[0]]:
        stage_result = train_stage_detector(prepared.windows_df, prepared.X, prepared.y, prepared.graph, prepared.cfg, n_folds=n_folds, seed=seed)
        stage_proba_full = scatter_stage_proba(prepared.windows_df, stage_result)
        seed_instances = build_forecast_instances(
            prepared.windows_df, prepared.split_map, horizon_multiple=fc_cfg["horizon_multiple"],
            max_seq_len=gru_cfg["max_seq_len"], drop_unconfirmed_negatives=fc_cfg["drop_unconfirmed_negatives"],
        )
        test_instances = [i for i in seed_instances if i.split == "test"]
        if test_instances:
            y_true_base = np.array([i.label for i in test_instances])
            max_stage2 = np.array([stage_proba_full[i.prefix_row_indices, lmep_idx].max() for i in test_instances])
            count_stage2 = np.array([float((stage_proba_full[i.prefix_row_indices, lmep_idx] > 0.5).sum()) for i in test_instances])
            count_stage2 = count_stage2 / max(count_stage2.max(), 1.0)
            for name, scores in [("Max-Stage2 (baseline)", max_stage2), ("Count-Stage2 (baseline)", count_stage2)]:
                m = compute_forecast_metrics(y_true_base, scores, capture_pcts)
                m["model"] = name
                m["seed"] = seed
                m["n_positive"] = int(y_true_base.sum())
                m["n_negative"] = int((y_true_base == 0).sum())
                rows.append(m)

    df = pd.DataFrame(rows)
    meta.stamp_df(df).to_csv(out_main / "impact_forecasting.csv", index=False)
    display_cols = ["model", "seed", "roc_auc", "pr_auc", "f1", "capture_at_1pct", "capture_at_2pct",
                    "capture_at_5pct", "capture_at_10pct", "n_positive", "n_negative"]
    display_cols = [c for c in display_cols if c in df.columns]
    df_to_latex_table(
        df, caption=f"{dataset}: impact forecasting (test).", label=f"tab:{dataset}_impact_forecasting",
        save_path=out_tables / f"tab_{dataset}_impact_forecasting.tex", columns=display_cols,
    )
    if curves:
        plot_pr_curves(curves, f"{dataset}: impact-forecast PR curve (test)", out_figs / "impact_forecast_pr_curve.png")
        cap_row = df[df["model"] == "Dual-stream GRU (main)"].iloc[0] if (df["model"] == "Dual-stream GRU (main)").any() else None
        if cap_row is not None:
            ks = [1, 2, 5, 10]
            vals = [cap_row.get(f"capture_at_{k}pct", np.nan) for k in ks]
            plot_sensitivity_line(ks, {"Capture@k": vals}, "k (%)", "Capture@k", f"{dataset}: Capture@k (test)",
                                  out_figs / "capture_at_k.png", x_is_categorical=True)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=["toniot", "edgeiiotset"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--allow-synthetic", action="store_true")
    args = ap.parse_args()

    require_real_data(args.dataset, allow_synthetic=args.allow_synthetic)

    set_global_seed(args.seeds[0])
    prepared = prepare_dataset(args.dataset)
    meta = RunMetadata.build(args.dataset, prepared.cfg, seed=args.seeds[0], is_real_data=True,
                             data_source_note="Real dataset verified against published class distributions; see MANUSCRIPT_RESULT_SUMMARY.md.")

    from src.utils.config import PROJECT_ROOT
    out_main = PROJECT_ROOT / "results" / args.dataset / "main"
    out_tables = PROJECT_ROOT / "results" / "manuscript_tables"
    out_figs = PROJECT_ROOT / "results" / "figures" / args.dataset
    for d in (out_main, out_tables, out_figs):
        d.mkdir(parents=True, exist_ok=True)

    section_dataset_summary(args.dataset, prepared, meta, out_main, out_tables)
    _, stage_results = section_stage_detection(args.dataset, prepared, args.seeds, args.n_folds, meta, out_main, out_tables, out_figs)
    section_one_vs_rest(args.dataset, stage_results, meta, out_main, out_tables, out_figs, args.seeds)
    section_impact_forecasting(args.dataset, prepared, args.seeds, args.n_folds, meta, out_main, out_tables, out_figs)

    logger.info(f"[{args.dataset}] Main results done. Written to {out_main}, {out_tables}, {out_figs}")


if __name__ == "__main__":
    main()
