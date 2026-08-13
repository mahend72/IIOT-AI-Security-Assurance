#!/usr/bin/env python3
"""Reviewer-response experiments for one dataset: window-size sensitivity,
forecast-horizon sensitivity, stage-mapping sensitivity, graph ablation,
meta-learner ablation, baseline-tuning fairness, asset-level bootstrap CIs,
and inference latency. Writes results/<dataset>/reviewer_experiments/*.csv
(+ SKIPPED_WITH_REASON JSON where an experiment is not valid for this
dataset) and results/manuscript_tables/tab_<dataset>_*.tex.

Refuses to run (SyntheticDataGuardError) if data/raw/<dataset>/ still has
the synthetic-generator marker -- see src/utils/data_provenance.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

from src.evaluation.bootstrap import asset_bootstrap_ci
from src.evaluation.forecast_metrics import capture_at_k, compute_forecast_metrics
from src.evaluation.forecast_validity import check_instance_level_validity
from src.evaluation.latex_tables import df_to_latex_table, write_skipped_latex_note
from src.evaluation.stage_metrics import compute_stage_metrics
from src.mapping.label_mapper import STAGE_ORDER
from src.models.gradient_boosting_models import build_lightgbm, build_xgboost
from src.models.random_forest_model import build_random_forest
from src.pipeline import prepare_dataset
from src.training.gcn_trainer import gcn_predict_proba, train_gcn
from src.training.impact_forecast_trainer import build_row_stream_features, train_impact_forecaster
from src.training.sequence_builder import build_forecast_instances
from src.training.stage_detector_trainer import scatter_stage_proba, train_stage_detector
from src.utils.config import load_dataset_config, load_stage_mapping_config, list_stage_mapping_variants
from src.utils.data_provenance import require_real_data
from src.utils.logging_utils import get_logger
from src.utils.run_metadata import RunMetadata
from src.utils.seed import set_global_seed

logger = get_logger("run_reviewer_experiments")
NUM_CLASSES = len(STAGE_ORDER)


def _macro_f1_and_per_class(y_true, y_pred):
    m = compute_stage_metrics(y_true, y_pred, STAGE_ORDER)
    out = {"macro_f1": m["macro_f1"]}
    for c in STAGE_ORDER:
        out[f"f1_{c.lower()}"] = m["per_class"][c]["f1"]
    return out


# ---------------------------------------------------------------------------
# 1. Window-size sensitivity
# ---------------------------------------------------------------------------
def window_sensitivity(dataset, deltas, n_folds, seed, out_dir, out_tables, meta_base):
    prepared0 = prepare_dataset(dataset)
    q = prepared0.bundle.quality
    if not q.get("window_sensitivity_valid", True):
        reasons = [q.get("window_sensitivity_invalid_reason", "window sensitivity not valid for this dataset.")]
        _write_skip(dataset, "window_sensitivity", reasons, out_dir, out_tables, meta_base)
        return None

    rows = []
    for dt in deltas:
        logger.info(f"[window sensitivity] Delta t = {dt}s")
        prepared = prepare_dataset(dataset, delta_t_seconds=dt)
        cfg = prepared.cfg
        result = train_stage_detector(prepared.windows_df, prepared.X, prepared.y, prepared.graph, cfg, n_folds=n_folds, seed=seed)
        y_pred = result.stacked_proba["test"].argmax(axis=1)
        metrics = _macro_f1_and_per_class(result.y_true["test"], y_pred)

        n_int = int(prepared.graph.edge_index_interaction.shape[1]) // 2
        n_temp = int(prepared.graph.edge_index_temporal.shape[1]) // 2
        n_nodes = len(prepared.graph.node_table)
        avg_degree = (2 * (n_int + n_temp)) / n_nodes if n_nodes else float("nan")

        row = {"delta_t_seconds": dt, "graph_nodes": n_nodes, "graph_edges": n_int + n_temp,
              "average_degree": round(avg_degree, 4), **metrics}

        fc_cfg = cfg["forecasting"]
        gru_cfg = cfg["models"]["gru_forecaster"]
        instances = build_forecast_instances(prepared.windows_df, prepared.split_map, fc_cfg["horizon_multiple"],
                                             gru_cfg["max_seq_len"], fc_cfg["drop_unconfirmed_negatives"])
        ok, _ = check_instance_level_validity(instances) if q.get("impact_forecasting_valid", True) else (False, [])
        if ok:
            stage_proba_full = scatter_stage_proba(prepared.windows_df, result)
            feat_iad, feat_lmep = build_row_stream_features(prepared.X, stage_proba_full)
            fres = train_impact_forecaster(instances, feat_iad, feat_lmep, cfg, seed=seed)
            fm = compute_forecast_metrics(fres.y_true["test"], fres.y_proba["test"], [5])
            row["pr_auc"] = fm.get("pr_auc", float("nan"))
            row["capture_at_5pct"] = fm.get("capture_at_5pct", float("nan"))
        else:
            row["pr_auc"] = float("nan")
            row["capture_at_5pct"] = float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    meta_base.stamp_df(df).to_csv(out_dir / "window_sensitivity.csv", index=False)
    df_to_latex_table(df, f"{dataset}: window-size sensitivity.", f"tab:{dataset}_window_sensitivity",
                      out_tables / f"tab_{dataset}_window_sensitivity.tex")
    return df


# ---------------------------------------------------------------------------
# 2. Forecast-horizon sensitivity
# ---------------------------------------------------------------------------
def horizon_sensitivity(dataset, horizon_multiples, n_folds, seed, out_dir, out_tables, meta_base):
    prepared = prepare_dataset(dataset)
    q = prepared.bundle.quality
    if not q.get("impact_forecasting_valid", True):
        _write_skip(dataset, "horizon_sensitivity", q.get("impact_forecasting_invalid_reasons", []), out_dir, out_tables, meta_base)
        return None

    stage_result = train_stage_detector(prepared.windows_df, prepared.X, prepared.y, prepared.graph, prepared.cfg, n_folds=n_folds, seed=seed)
    stage_proba_full = scatter_stage_proba(prepared.windows_df, stage_result)
    feat_iad, feat_lmep = build_row_stream_features(prepared.X, stage_proba_full)
    gru_cfg = prepared.cfg["models"]["gru_forecaster"]
    fc_cfg = prepared.cfg["forecasting"]

    rows = []
    all_invalid_reasons = []
    for h in horizon_multiples:
        instances = build_forecast_instances(prepared.windows_df, prepared.split_map, h, gru_cfg["max_seq_len"], fc_cfg["drop_unconfirmed_negatives"])
        ok, reasons = check_instance_level_validity(instances)
        if not ok:
            all_invalid_reasons.append(f"H={h}: {'; '.join(reasons)}")
            continue
        fres = train_impact_forecaster(instances, feat_iad, feat_lmep, prepared.cfg, seed=seed)
        m = compute_forecast_metrics(fres.y_true["test"], fres.y_proba["test"], [1, 2, 5, 10])
        rows.append({"horizon_multiple": h, "roc_auc": m.get("roc_auc"), "pr_auc": m.get("pr_auc"),
                    "capture_at_1pct": m.get("capture_at_1pct"), "capture_at_2pct": m.get("capture_at_2pct"),
                    "capture_at_5pct": m.get("capture_at_5pct"), "capture_at_10pct": m.get("capture_at_10pct"),
                    "n_positive": int(fres.y_true["test"].sum())})

    if not rows:
        _write_skip(dataset, "horizon_sensitivity",
                    [f"instance-level gate failed at every candidate horizon: {'; '.join(all_invalid_reasons)}"],
                    out_dir, out_tables, meta_base)
        return None

    df = pd.DataFrame(rows)
    meta_base.stamp_df(df).to_csv(out_dir / "horizon_sensitivity.csv", index=False)
    df_to_latex_table(df, f"{dataset}: forecast-horizon sensitivity.", f"tab:{dataset}_horizon_sensitivity",
                      out_tables / f"tab_{dataset}_horizon_sensitivity.tex")
    return df


# ---------------------------------------------------------------------------
# 3. Stage-mapping sensitivity
# ---------------------------------------------------------------------------
def stage_mapping_sensitivity(dataset, n_folds, seed, out_dir, out_tables, meta_base):
    variants = list_stage_mapping_variants(dataset)
    rows = []
    for variant_name in variants:
        logger.info(f"[stage-mapping sensitivity] variant = {variant_name}")
        prepared = prepare_dataset(dataset, mapping_variant=variant_name)
        result = train_stage_detector(prepared.windows_df, prepared.X, prepared.y, prepared.graph, prepared.cfg, n_folds=n_folds, seed=seed)
        y_pred = result.stacked_proba["test"].argmax(axis=1)
        metrics = _macro_f1_and_per_class(result.y_true["test"], y_pred)
        row = {"stage_mapping_variant": variant_name, **metrics}

        q = prepared.bundle.quality
        if q.get("impact_forecasting_valid", True):
            fc_cfg = prepared.cfg["forecasting"]
            gru_cfg = prepared.cfg["models"]["gru_forecaster"]
            instances = build_forecast_instances(prepared.windows_df, prepared.split_map, fc_cfg["horizon_multiple"], gru_cfg["max_seq_len"], fc_cfg["drop_unconfirmed_negatives"])
            ok, _ = check_instance_level_validity(instances)
            if ok:
                stage_proba_full = scatter_stage_proba(prepared.windows_df, result)
                feat_iad, feat_lmep = build_row_stream_features(prepared.X, stage_proba_full)
                fres = train_impact_forecaster(instances, feat_iad, feat_lmep, prepared.cfg, seed=seed)
                fm = compute_forecast_metrics(fres.y_true["test"], fres.y_proba["test"], [5])
                row["pr_auc"] = fm.get("pr_auc", float("nan"))
                row["capture_at_5pct"] = fm.get("capture_at_5pct", float("nan"))
            else:
                row["pr_auc"], row["capture_at_5pct"] = float("nan"), float("nan")
        else:
            row["pr_auc"], row["capture_at_5pct"] = float("nan"), float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    meta_base.stamp_df(df).to_csv(out_dir / "stage_mapping_sensitivity.csv", index=False)
    df_to_latex_table(
        df, f"{dataset}: stage-mapping sensitivity.", f"tab:{dataset}_stage_mapping_sensitivity",
        out_tables / f"tab_{dataset}_stage_mapping_sensitivity.tex",
        columns=["stage_mapping_variant", "macro_f1", "f1_iad", "f1_lmep", "f1_imp", "pr_auc", "capture_at_5pct"],
        column_headers=["Mapping", "Macro-F1", "F1(IAD)", "F1(LMEP)", "F1(IMP)", "PR-AUC", "Capture@5\\%"],
    )
    return df


# ---------------------------------------------------------------------------
# 4. Graph ablation
# ---------------------------------------------------------------------------
def graph_ablation(dataset, n_folds, seed, out_dir, out_tables, meta_base):
    prepared = prepare_dataset(dataset)
    rows = []
    variants = ["interaction_only", "temporal_only", "both"]
    for mode in variants:
        logger.info(f"[graph ablation] mode = {mode}")
        t0 = time.perf_counter()
        result = train_stage_detector(prepared.windows_df, prepared.X, prepared.y, prepared.graph, prepared.cfg, graph_mode=mode, n_folds=n_folds, seed=seed)
        train_time = time.perf_counter() - t0

        data_full = prepared.graph.to_pyg_data(prepared.X, prepared.y, mode=mode)
        n_nodes = data_full.num_nodes
        n_edges = data_full.edge_index.shape[1] // 2
        avg_degree = (2 * n_edges) / n_nodes if n_nodes else float("nan")

        t0 = time.perf_counter()
        for _ in range(20):
            gcn_predict_proba(result.gcn_model, data_full)
        inf_latency_ms = (time.perf_counter() - t0) / 20 * 1000

        y_pred = result.stacked_proba["test"].argmax(axis=1)
        metrics = _macro_f1_and_per_class(result.y_true["test"], y_pred)
        rows.append({"graph_mode": mode, "nodes": n_nodes, "edges": n_edges, "average_degree": round(avg_degree, 4),
                    **metrics, "training_time_s": round(train_time, 3), "inference_latency_ms": round(inf_latency_ms, 4)})

    note = ("Directed-communication and typed/relational edge variants are not implemented in this codebase "
           "(the GNN backbones used -- GCN/GraphSAGE/GAT -- are undirected, untyped message-passing layers); "
           "only interaction-only / temporal-only / both are reported.")
    df = pd.DataFrame(rows)
    meta_base.stamp_df(df).to_csv(out_dir / "graph_ablation.csv", index=False)
    df_to_latex_table(df, f"{dataset}: graph ablation.", f"tab:{dataset}_graph_ablation",
                      out_tables / f"tab_{dataset}_graph_ablation.tex", note=note)
    return df


# ---------------------------------------------------------------------------
# 5. Meta-learner ablation
# ---------------------------------------------------------------------------
def meta_learner_ablation(dataset, n_folds, seed, out_dir, out_tables, meta_base):
    prepared = prepare_dataset(dataset)
    rows = []
    for meta_type in ["logistic_regression", "mlp", "gradient_boosting"]:
        logger.info(f"[meta-learner ablation] type = {meta_type}")
        t0 = time.perf_counter()
        result = train_stage_detector(prepared.windows_df, prepared.X, prepared.y, prepared.graph, prepared.cfg, n_folds=n_folds, seed=seed, meta_type=meta_type)
        train_time = time.perf_counter() - t0
        y_pred = result.stacked_proba["test"].argmax(axis=1)
        metrics = _macro_f1_and_per_class(result.y_true["test"], y_pred)

        # Calibration: mean one-vs-rest Brier score across classes (lower = better calibrated).
        y_true_oh = np.eye(NUM_CLASSES)[result.y_true["test"]]
        briers = [brier_score_loss(y_true_oh[:, c], result.stacked_proba["test"][:, c]) for c in range(NUM_CLASSES)]
        rows.append({"meta_learner": meta_type, **metrics, "mean_brier_score": float(np.mean(briers)),
                    "training_time_s": round(train_time, 3)})

    df = pd.DataFrame(rows)
    meta_base.stamp_df(df).to_csv(out_dir / "meta_learner_ablation.csv", index=False)
    df_to_latex_table(df, f"{dataset}: meta-learner ablation.", f"tab:{dataset}_meta_learner_ablation",
                      out_tables / f"tab_{dataset}_meta_learner_ablation.tex")
    return df


# ---------------------------------------------------------------------------
# 6. Baseline tuning fairness
# ---------------------------------------------------------------------------
def baseline_tuning(dataset, seed, out_dir, out_tables, meta_base):
    prepared = prepare_dataset(dataset)
    split = prepared.windows_df["split"].to_numpy()
    train_mask, val_mask, test_mask = split == "train", split == "val", split == "test"
    X, y = prepared.X, prepared.y

    def val_test_f1(model):
        val_f1 = f1_score(y[val_mask], model.predict(X[val_mask]), average="macro", zero_division=0)
        test_f1 = f1_score(y[test_mask], model.predict(X[test_mask]), average="macro", zero_division=0)
        return val_f1, test_f1

    rows = []

    rf_grid = [{"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
              {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2},
              {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 2},
              {"n_estimators": 500, "max_depth": 20, "min_samples_leaf": 4}]
    best = {"val": -1}
    for params in rf_grid:
        m = build_random_forest({**prepared.cfg["models"]["random_forest"], **params})
        m.fit(X[train_mask], y[train_mask])
        v, t = val_test_f1(m)
        if v > best["val"]:
            best = {"val": v, "test": t, "params": params}
    rows.append({"model": "RF", "search_space": str(rf_grid), "selected_hyperparameters": str(best["params"]),
                "validation_metric": "macro_f1", "val_score": best["val"], "test_score": best["test"]})

    xgb_grid = [{"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1},
               {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
               {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05}]
    best = {"val": -1}
    for params in xgb_grid:
        m = build_xgboost({**prepared.cfg["models"]["xgboost"], **params}, NUM_CLASSES)
        m.fit(X[train_mask], y[train_mask])
        v, t = val_test_f1(m)
        if v > best["val"]:
            best = {"val": v, "test": t, "params": params}
    rows.append({"model": "XGBoost", "search_space": str(xgb_grid), "selected_hyperparameters": str(best["params"]),
                "validation_metric": "macro_f1", "val_score": best["val"], "test_score": best["test"]})

    lgb_grid = [{"n_estimators": 100, "max_depth": -1, "learning_rate": 0.1},
               {"n_estimators": 300, "max_depth": -1, "learning_rate": 0.1},
               {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.05}]
    best = {"val": -1}
    for params in lgb_grid:
        m = build_lightgbm({**prepared.cfg["models"]["lightgbm"], **params}, NUM_CLASSES)
        m.fit(X[train_mask], y[train_mask])
        v, t = val_test_f1(m)
        if v > best["val"]:
            best = {"val": v, "test": t, "params": params}
    rows.append({"model": "LightGBM", "search_space": str(lgb_grid), "selected_hyperparameters": str(best["params"]),
                "validation_metric": "macro_f1", "val_score": best["val"], "test_score": best["test"]})

    gnn_grid = [{"hidden_dim": 32, "lr": 0.005}, {"hidden_dim": 64, "lr": 0.005}, {"hidden_dim": 64, "lr": 0.01}]
    for name, conv in [("GCN", "gcn"), ("GraphSAGE", "sage"), ("GAT", "gat")]:
        data_full = prepared.graph.to_pyg_data(X, y, mode="both")
        best = {"val": -1}
        for params in gnn_grid:
            cfg = {**prepared.cfg["models"]["gcn"], **params}
            model, _ = train_gcn(data_full, data_full.train_mask, data_full.val_mask, NUM_CLASSES, cfg, seed=seed, conv_type=conv)
            proba = gcn_predict_proba(model, data_full)
            v = f1_score(y[val_mask], proba[val_mask].argmax(axis=1), average="macro", zero_division=0)
            t = f1_score(y[test_mask], proba[test_mask].argmax(axis=1), average="macro", zero_division=0)
            if v > best["val"]:
                best = {"val": v, "test": t, "params": params}
        rows.append({"model": name, "search_space": str(gnn_grid), "selected_hyperparameters": str(best["params"]),
                    "validation_metric": "macro_f1", "val_score": best["val"], "test_score": best["test"]})

    df = pd.DataFrame(rows)
    meta_base.stamp_df(df).to_csv(out_dir / "baseline_tuning.csv", index=False)
    df_to_latex_table(df, f"{dataset}: baseline-tuning fairness (selected on VAL only).",
                      f"tab:{dataset}_baseline_tuning", out_tables / f"tab_{dataset}_baseline_tuning.tex")
    return df


# ---------------------------------------------------------------------------
# 7. Bootstrap confidence intervals (asset-level resampling)
# ---------------------------------------------------------------------------
def bootstrap_confidence_intervals(dataset, n_folds, seed, out_dir, out_tables, meta_base, n_iterations=1000):
    prepared = prepare_dataset(dataset)
    result = train_stage_detector(prepared.windows_df, prepared.X, prepared.y, prepared.graph, prepared.cfg, n_folds=n_folds, seed=seed)
    test_mask = prepared.windows_df["split"].to_numpy() == "test"
    asset_ids_test = prepared.windows_df.loc[test_mask, "asset_id"].to_numpy()
    y_true_test = result.y_true["test"]
    y_pred_test = result.stacked_proba["test"].argmax(axis=1)

    def macro_f1_fn(yt, yp):
        return f1_score(yt, yp, average="macro", zero_division=0)

    rows = []
    ci_macro = asset_bootstrap_ci(macro_f1_fn, y_true_test, y_pred_test, asset_ids_test, n_iterations=n_iterations, seed=seed)
    rows.append({"metric": "macro_f1", **ci_macro})

    q = prepared.bundle.quality
    if q.get("impact_forecasting_valid", True):
        fc_cfg = prepared.cfg["forecasting"]
        gru_cfg = prepared.cfg["models"]["gru_forecaster"]
        instances = build_forecast_instances(prepared.windows_df, prepared.split_map, fc_cfg["horizon_multiple"], gru_cfg["max_seq_len"], fc_cfg["drop_unconfirmed_negatives"])
        ok, _ = check_instance_level_validity(instances)
        if ok:
            stage_proba_full = scatter_stage_proba(prepared.windows_df, result)
            feat_iad, feat_lmep = build_row_stream_features(prepared.X, stage_proba_full)
            fres = train_impact_forecaster(instances, feat_iad, feat_lmep, prepared.cfg, seed=seed)
            fc_asset_ids = fres.asset_ids["test"]
            if len(fres.y_true["test"]) and len(set(fres.y_true["test"].tolist())) > 1:
                ci_prauc = asset_bootstrap_ci(average_precision_score, fres.y_true["test"], fres.y_proba["test"], fc_asset_ids, n_iterations=n_iterations, seed=seed)
                rows.append({"metric": "pr_auc", **ci_prauc})
                for k in (5, 10):
                    fn = lambda yt, ys, kk=k: capture_at_k(yt, ys, kk)
                    ci_cap = asset_bootstrap_ci(fn, fres.y_true["test"], fres.y_proba["test"], fc_asset_ids, n_iterations=n_iterations, seed=seed)
                    rows.append({"metric": f"capture_at_{k}pct", **ci_cap})

    df = pd.DataFrame(rows)
    meta_base.stamp_df(df).to_csv(out_dir / "bootstrap_confidence_intervals.csv", index=False)
    df_to_latex_table(df, f"{dataset}: 95\\% bootstrap CIs (asset-level resampling).",
                      f"tab:{dataset}_bootstrap_ci", out_tables / f"tab_{dataset}_bootstrap_ci.tex",
                      columns=["metric", "point", "lower", "upper", "n_unique_assets"])
    return df


# ---------------------------------------------------------------------------
# 8. Inference latency
# ---------------------------------------------------------------------------
def inference_latency(dataset, n_folds, seed, out_dir, out_tables, meta_base, n_repeats=30):
    prepared = prepare_dataset(dataset)
    stage_result = train_stage_detector(prepared.windows_df, prepared.X, prepared.y, prepared.graph, prepared.cfg, n_folds=n_folds, seed=seed)
    split = prepared.windows_df["split"].to_numpy()
    test_mask = split == "test"
    X_test = prepared.X[test_mask]
    data_full = prepared.graph.to_pyg_data(prepared.X, prepared.y, mode="both")

    rows = []

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        prepared.feature_builder.transform(prepared.windows_df.iloc[:1])
    feat_ms = (time.perf_counter() - t0) / n_repeats * 1000
    rows.append({"component": "feature_aggregation (per instance)", "latency_ms": feat_ms})

    x0 = X_test[:1]
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        stage_result.rf_model.predict_proba(x0)
    rf_ms = (time.perf_counter() - t0) / n_repeats * 1000
    rows.append({"component": "RF scoring (per instance)", "latency_ms": rf_ms})

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        gcn_predict_proba(stage_result.gcn_model, data_full)
    gcn_full_ms = (time.perf_counter() - t0) / n_repeats * 1000
    gcn_per_node_ms = gcn_full_ms / max(int(test_mask.sum()), 1)
    rows.append({"component": "GCN full-graph forward pass", "latency_ms": gcn_full_ms})
    rows.append({"component": "GCN node scoring (amortized per test node)", "latency_ms": gcn_per_node_ms})

    rf_p = stage_result.rf_proba["test"][:1]
    gcn_p = stage_result.gcn_proba["test"][:1]
    meta_feat = np.concatenate([rf_p, gcn_p], axis=1)
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        stage_result.meta_learner.predict_proba(meta_feat)
    meta_ms = (time.perf_counter() - t0) / n_repeats * 1000
    rows.append({"component": "Stacking meta-learner (per instance)", "latency_ms": meta_ms})

    end_to_end_ms = feat_ms + rf_ms + gcn_per_node_ms + meta_ms
    q = prepared.bundle.quality
    if q.get("impact_forecasting_valid", True):
        fc_cfg = prepared.cfg["forecasting"]
        gru_cfg = prepared.cfg["models"]["gru_forecaster"]
        instances = build_forecast_instances(prepared.windows_df, prepared.split_map, fc_cfg["horizon_multiple"], gru_cfg["max_seq_len"], fc_cfg["drop_unconfirmed_negatives"])
        ok, _ = check_instance_level_validity(instances)
        if ok:
            stage_proba_full = scatter_stage_proba(prepared.windows_df, stage_result)
            feat_iad, feat_lmep = build_row_stream_features(prepared.X, stage_proba_full)
            fres = train_impact_forecaster(instances, feat_iad, feat_lmep, prepared.cfg, seed=seed)
            inst = instances[0]
            t = len(inst.prefix_row_indices)
            x_iad = torch.from_numpy(feat_iad[inst.prefix_row_indices]).unsqueeze(0).float()
            x_lmep = torch.from_numpy(feat_lmep[inst.prefix_row_indices]).unsqueeze(0).float()
            lengths = torch.tensor([t])
            with torch.no_grad():
                t0 = time.perf_counter()
                for _ in range(n_repeats):
                    fres.model(x_iad, x_lmep, lengths)
                gru_ms = (time.perf_counter() - t0) / n_repeats * 1000
            rows.append({"component": "GRU impact scoring (per instance)", "latency_ms": gru_ms})
            end_to_end_ms += gru_ms

    rows.append({"component": "End-to-end (per instance, all applicable stages)", "latency_ms": end_to_end_ms})
    rows.append({"component": "Throughput (instances/sec, end-to-end)", "latency_ms": 1000.0 / end_to_end_ms if end_to_end_ms > 0 else float("nan")})

    df = pd.DataFrame(rows)
    meta_base.stamp_df(df).to_csv(out_dir / "inference_latency.csv", index=False)
    df_to_latex_table(df, f"{dataset}: inference latency.", f"tab:{dataset}_inference_latency",
                      out_tables / f"tab_{dataset}_inference_latency.tex")
    return df


def _write_skip(dataset, experiment, reasons, out_dir, out_tables, meta_base):
    report = meta_base.stamp_json({"dataset": dataset, "experiment": experiment, "run": False, "reasons": reasons})
    (out_dir / f"{experiment}_SKIPPED_WITH_REASON.json").write_text(json.dumps(report, indent=2))
    write_skipped_latex_note(f"{dataset}: {experiment} -- SKIPPED.", f"tab:{dataset}_{experiment}",
                             out_tables / f"tab_{dataset}_{experiment}.tex", reasons, dataset)
    logger.warning(f"[{dataset}] {experiment} SKIPPED: {reasons}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=["toniot", "edgeiiotset"])
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap-iterations", type=int, default=1000)
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--only", nargs="+", default=None,
                    help="restrict to a subset of {window,horizon,mapping,graph,meta,tuning,bootstrap,latency}")
    args = ap.parse_args()

    require_real_data(args.dataset, allow_synthetic=args.allow_synthetic)
    set_global_seed(args.seed)

    from src.utils.config import PROJECT_ROOT
    out_dir = PROJECT_ROOT / "results" / args.dataset / "reviewer_experiments"
    out_tables = PROJECT_ROOT / "results" / "manuscript_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    dataset_cfg = load_dataset_config(args.dataset)
    meta_base = RunMetadata.build(args.dataset, dataset_cfg, seed=args.seed)

    def want(name):
        return args.only is None or name in args.only

    if want("window"):
        window_sensitivity(args.dataset, dataset_cfg["window"]["candidate_delta_t_seconds"], args.n_folds, args.seed, out_dir, out_tables, meta_base)
    if want("horizon"):
        horizon_sensitivity(args.dataset, dataset_cfg["forecasting"]["candidate_horizon_multiples"], args.n_folds, args.seed, out_dir, out_tables, meta_base)
    if want("mapping"):
        stage_mapping_sensitivity(args.dataset, args.n_folds, args.seed, out_dir, out_tables, meta_base)
    if want("graph"):
        graph_ablation(args.dataset, args.n_folds, args.seed, out_dir, out_tables, meta_base)
    if want("meta"):
        meta_learner_ablation(args.dataset, args.n_folds, args.seed, out_dir, out_tables, meta_base)
    if want("tuning"):
        baseline_tuning(args.dataset, args.seed, out_dir, out_tables, meta_base)
    if want("bootstrap"):
        bootstrap_confidence_intervals(args.dataset, args.n_folds, args.seed, out_dir, out_tables, meta_base, n_iterations=args.bootstrap_iterations)
    if want("latency"):
        inference_latency(args.dataset, args.n_folds, args.seed, out_dir, out_tables, meta_base)

    logger.info(f"[{args.dataset}] Reviewer experiments done. Written to {out_dir}, {out_tables}")


if __name__ == "__main__":
    main()
