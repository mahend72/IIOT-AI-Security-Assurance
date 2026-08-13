#!/usr/bin/env python3
"""Reviewer-response experiments for one dataset:
  - window sensitivity      (Delta t in {30, 60, 120, 300}s)
  - horizon sensitivity      (H in {10, 30, 60} x Delta t)
  - graph ablation           (interaction-only / temporal-only / both)
  - meta-learner ablation    (logistic regression / MLP / gradient boosting)
  - baseline (RF) tuning table -- selected on the VAL split only, never test
  - inference latency table

All results are written to results/<dataset>/sensitivity/. Figures use the
shared palette (src/evaluation/plotting.py) so every series is identified
by a fixed color across the whole project.

Example:
    python scripts/run_sensitivity.py --dataset toniot
    python scripts/run_sensitivity.py --dataset toniot --skip-tuning --skip-latency
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from src.evaluation.plotting import plot_sensitivity_line
from src.evaluation.report import results_dir, save_csv, save_json
from src.evaluation.forecast_metrics import compute_forecast_metrics
from src.models.random_forest_model import build_random_forest
from src.pipeline import prepare_dataset
from src.training.gcn_trainer import gcn_predict_proba
from src.training.impact_forecast_trainer import build_row_stream_features, train_impact_forecaster
from src.training.sequence_builder import build_forecast_instances
from src.training.stage_detector_trainer import scatter_stage_proba, train_stage_detector
from src.utils.config import load_dataset_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_global_seed

logger = get_logger("run_sensitivity")


def window_sensitivity(dataset: str, deltas, n_folds: int, seed: int, out_dir: Path):
    rows = []
    for dt in deltas:
        logger.info(f"[window sensitivity] Delta t = {dt}s")
        prepared = prepare_dataset(dataset, delta_t_seconds=dt)
        cfg = copy.deepcopy(prepared.cfg)
        cfg["models"]["gcn"]["epochs"] = min(cfg["models"]["gcn"]["epochs"], 60)
        result = train_stage_detector(prepared.windows_df, prepared.X, prepared.y, prepared.graph, cfg, n_folds=n_folds, seed=seed)
        for model_name, proba in [("RF", result.rf_proba), ("GCN", result.gcn_proba), ("Stacked", result.stacked_proba)]:
            f1 = f1_score(result.y_true["test"], proba["test"].argmax(axis=1), average="macro", zero_division=0)
            rows.append({"delta_t_seconds": dt, "model": model_name, "macro_f1_test": f1, "n_windows": len(prepared.windows_df)})
    df = pd.DataFrame(rows)
    save_csv(df, out_dir / "window_sensitivity.csv")
    pivot = df.pivot(index="delta_t_seconds", columns="model", values="macro_f1_test")
    plot_sensitivity_line(
        list(pivot.index), {c: pivot[c].tolist() for c in pivot.columns},
        "Window size Delta t (s)", "Macro-F1 (test)", f"{dataset}: window-size sensitivity",
        out_dir / "window_sensitivity.png",
    )
    return df


def horizon_sensitivity(dataset: str, horizon_multiples, n_folds: int, seed: int, out_dir: Path):
    prepared = prepare_dataset(dataset)
    quality = prepared.bundle.quality
    if quality.get("impact_forecasting_valid", True) is False:
        logger.warning(f"[{dataset}] Skipping horizon sensitivity: impact forecasting data-quality gate failed "
                        f"({quality.get('impact_forecasting_invalid_reasons')}).")
        return None

    stage_result = train_stage_detector(prepared.windows_df, prepared.X, prepared.y, prepared.graph, prepared.cfg, n_folds=n_folds, seed=seed)
    stage_proba_full = scatter_stage_proba(prepared.windows_df, stage_result)
    feat_iad, feat_lmep = build_row_stream_features(prepared.X, stage_proba_full)

    gru_cfg = prepared.cfg["models"]["gru_forecaster"]
    fc_cfg = prepared.cfg["forecasting"]
    rows = []
    for h in horizon_multiples:
        logger.info(f"[horizon sensitivity] H = {h} x Delta t")
        instances = build_forecast_instances(
            prepared.windows_df, prepared.split_map, horizon_multiple=h,
            max_seq_len=gru_cfg["max_seq_len"], drop_unconfirmed_negatives=fc_cfg["drop_unconfirmed_negatives"],
        )
        if sum(1 for i in instances if i.split == "test") < 5:
            logger.warning(f"  too few test instances at H={h}, skipping")
            continue
        fres = train_impact_forecaster(instances, feat_iad, feat_lmep, prepared.cfg, seed=seed)
        m = compute_forecast_metrics(fres.y_true["test"], fres.y_proba["test"], prepared.cfg["evaluation"]["capture_at_percents"])
        rows.append({"horizon_multiple": h, **{k: v for k, v in m.items() if not isinstance(v, dict)}})
    df = pd.DataFrame(rows)
    save_csv(df, out_dir / "horizon_sensitivity.csv")
    if not df.empty:
        plot_sensitivity_line(
            df["horizon_multiple"].tolist(), {"ROC-AUC": df["roc_auc"].tolist(), "PR-AUC": df["pr_auc"].tolist()},
            "Horizon H (x Delta t)", "Score (test)", f"{dataset}: forecasting-horizon sensitivity",
            out_dir / "horizon_sensitivity.png",
        )
    return df


def graph_ablation(dataset: str, n_folds: int, seed: int, out_dir: Path):
    prepared = prepare_dataset(dataset)
    rows = []
    for mode in ["interaction_only", "temporal_only", "both"]:
        logger.info(f"[graph ablation] mode = {mode}")
        result = train_stage_detector(prepared.windows_df, prepared.X, prepared.y, prepared.graph, prepared.cfg, graph_mode=mode, n_folds=n_folds, seed=seed)
        for model_name, proba in [("GCN", result.gcn_proba), ("Stacked", result.stacked_proba)]:
            f1 = f1_score(result.y_true["test"], proba["test"].argmax(axis=1), average="macro", zero_division=0)
            rows.append({"graph_mode": mode, "model": model_name, "macro_f1_test": f1})
    df = pd.DataFrame(rows)
    save_csv(df, out_dir / "graph_ablation.csv")
    pivot = df.pivot(index="graph_mode", columns="model", values="macro_f1_test")
    plot_sensitivity_line(
        list(pivot.index), {c: pivot[c].tolist() for c in pivot.columns},
        "Graph construction mode", "Macro-F1 (test)", f"{dataset}: graph-ablation",
        out_dir / "graph_ablation.png",
    )
    return df


def meta_learner_ablation(dataset: str, n_folds: int, seed: int, out_dir: Path):
    prepared = prepare_dataset(dataset)
    rows = []
    for meta_type in ["logistic_regression", "mlp", "gradient_boosting"]:
        logger.info(f"[meta-learner ablation] type = {meta_type}")
        result = train_stage_detector(prepared.windows_df, prepared.X, prepared.y, prepared.graph, prepared.cfg, n_folds=n_folds, seed=seed, meta_type=meta_type)
        f1 = f1_score(result.y_true["test"], result.stacked_proba["test"].argmax(axis=1), average="macro", zero_division=0)
        rows.append({"meta_learner": meta_type, "macro_f1_test": f1})
    df = pd.DataFrame(rows)
    save_csv(df, out_dir / "meta_learner_ablation.csv")
    plot_sensitivity_line(
        df["meta_learner"].tolist(), {"Stacked": df["macro_f1_test"].tolist()},
        "Meta-learner", "Macro-F1 (test)", f"{dataset}: meta-learner ablation",
        out_dir / "meta_learner_ablation.png",
    )
    return df


def baseline_tuning_table(dataset: str, out_dir: Path):
    """Small RF hyperparameter grid, selected on VAL macro-F1 only -- test is
    never touched until the final run_stage_detection.py report."""
    prepared = prepare_dataset(dataset)
    grid = [
        {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2},
        {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 2},
        {"n_estimators": 500, "max_depth": 20, "min_samples_leaf": 4},
    ]
    split = prepared.windows_df["split"].to_numpy()
    train_mask, val_mask = split == "train", split == "val"
    rows = []
    for params in grid:
        rf = build_random_forest({**prepared.cfg["models"]["random_forest"], **params})
        rf.fit(prepared.X[train_mask], prepared.y[train_mask])
        val_f1 = f1_score(prepared.y[val_mask], rf.predict(prepared.X[val_mask]), average="macro", zero_division=0)
        rows.append({**params, "val_macro_f1": val_f1})
    df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    save_csv(df, out_dir / "baseline_rf_tuning.csv")
    logger.info(f"[{dataset}] Best RF config by VAL macro-F1: {df.iloc[0].to_dict()}")
    return df


def inference_latency_table(dataset: str, n_folds: int, seed: int, out_dir: Path, n_repeats: int = 50):
    prepared = prepare_dataset(dataset)
    stage_result = train_stage_detector(prepared.windows_df, prepared.X, prepared.y, prepared.graph, prepared.cfg, n_folds=n_folds, seed=seed)
    split = prepared.windows_df["split"].to_numpy()
    test_mask = split == "test"
    X_test = prepared.X[test_mask]
    data_full = prepared.graph.to_pyg_data(prepared.X, prepared.y, mode="both")

    rows = []

    # RF: single-instance predict latency.
    x0 = X_test[:1]
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        stage_result.rf_model.predict_proba(x0)
    rf_ms = (time.perf_counter() - t0) / n_repeats * 1000
    rows.append({"component": "RF (per instance)", "latency_ms": rf_ms})

    # GCN: full-graph forward pass, amortized per test node (transductive
    # inference recomputes the whole graph, so we report both the whole-pass
    # cost and the per-node amortized cost).
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        gcn_predict_proba(stage_result.gcn_model, data_full)
    gcn_full_ms = (time.perf_counter() - t0) / n_repeats * 1000
    rows.append({"component": "GCN (full-graph forward pass)", "latency_ms": gcn_full_ms})
    rows.append({"component": "GCN (amortized per test node)", "latency_ms": gcn_full_ms / max(int(test_mask.sum()), 1)})

    # Stacked meta-learner: negligible extra cost on top of RF+GCN proba.
    rf_p = stage_result.rf_proba["test"][:1]
    gcn_p = stage_result.gcn_proba["test"][:1]
    meta_feat = np.concatenate([rf_p, gcn_p], axis=1)
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        stage_result.meta_learner.predict_proba(meta_feat)
    meta_ms = (time.perf_counter() - t0) / n_repeats * 1000
    rows.append({"component": "Meta-learner (per instance, given RF+GCN proba)", "latency_ms": meta_ms})
    rows.append({"component": "Stacked total (RF + GCN full-pass + meta, per instance amortized)",
                  "latency_ms": rf_ms + gcn_full_ms / max(int(test_mask.sum()), 1) + meta_ms})

    # GRU forecaster, if valid for this dataset.
    if prepared.bundle.quality.get("impact_forecasting_valid", True):
        stage_proba_full = scatter_stage_proba(prepared.windows_df, stage_result)
        feat_iad, feat_lmep = build_row_stream_features(prepared.X, stage_proba_full)
        fc_cfg = prepared.cfg["forecasting"]
        gru_cfg = prepared.cfg["models"]["gru_forecaster"]
        instances = build_forecast_instances(prepared.windows_df, prepared.split_map, fc_cfg["horizon_multiple"], gru_cfg["max_seq_len"], fc_cfg["drop_unconfirmed_negatives"])
        if instances:
            fres = train_impact_forecaster(instances, feat_iad, feat_lmep, prepared.cfg, seed=seed)
            model = fres.model
            inst = instances[0]
            t = len(inst.prefix_row_indices)
            x_iad = torch.from_numpy(feat_iad[inst.prefix_row_indices]).unsqueeze(0).float()
            x_lmep = torch.from_numpy(feat_lmep[inst.prefix_row_indices]).unsqueeze(0).float()
            lengths = torch.tensor([t])
            with torch.no_grad():
                t0 = time.perf_counter()
                for _ in range(n_repeats):
                    model(x_iad, x_lmep, lengths)
                gru_ms = (time.perf_counter() - t0) / n_repeats * 1000
            rows.append({"component": "Dual-stream GRU forecaster (per instance)", "latency_ms": gru_ms})

    df = pd.DataFrame(rows)
    save_csv(df, out_dir / "inference_latency.csv")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=["toniot", "edgeiiotset"])
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-tuning", action="store_true")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--skip-window", action="store_true")
    ap.add_argument("--skip-horizon", action="store_true")
    ap.add_argument("--skip-graph-ablation", action="store_true")
    ap.add_argument("--skip-meta-ablation", action="store_true")
    args = ap.parse_args()

    set_global_seed(args.seed)
    out_dir = results_dir(args.dataset, "sensitivity")
    summary = {}
    dataset_cfg = load_dataset_config(args.dataset)

    if not args.skip_window:
        summary["window_sensitivity"] = "ok"
        window_sensitivity(args.dataset, dataset_cfg["window"]["candidate_delta_t_seconds"], args.n_folds, args.seed, out_dir)
    if not args.skip_horizon:
        r = horizon_sensitivity(args.dataset, dataset_cfg["forecasting"]["candidate_horizon_multiples"], args.n_folds, args.seed, out_dir)
        summary["horizon_sensitivity"] = "ok" if r is not None else "skipped (data quality gate)"
    if not args.skip_graph_ablation:
        graph_ablation(args.dataset, args.n_folds, args.seed, out_dir)
        summary["graph_ablation"] = "ok"
    if not args.skip_meta_ablation:
        meta_learner_ablation(args.dataset, args.n_folds, args.seed, out_dir)
        summary["meta_learner_ablation"] = "ok"
    if not args.skip_tuning:
        baseline_tuning_table(args.dataset, out_dir)
        summary["baseline_tuning"] = "ok"
    if not args.skip_latency:
        inference_latency_table(args.dataset, args.n_folds, args.seed, out_dir)
        summary["inference_latency"] = "ok"

    save_json(summary, out_dir / "sensitivity_run_summary.json")
    logger.info(f"[{args.dataset}] Sensitivity experiments done: {summary}")
    logger.info(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
