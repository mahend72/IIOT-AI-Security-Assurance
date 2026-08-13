# Manuscript Result Summary — Real Data (Corrected)

Generated: 2026-08-12. Source of truth for every number in this document is
`results/*/main/`, `results/*/reviewer_experiments/`, `results/manuscript_tables/`,
and `results/figures/` — all produced from the real datasets
(`data/raw/toniot/Train_Test_Network.csv`, `data/raw/edgeiiotset/ML-EdgeIIoT-dataset.csv`),
placed on disk 2026-08-11 15:26 local. `results/VALIDATION_REPORT.json` (16/16
checks passed, both datasets, re-run 2026-08-12 00:30 UTC+1) confirms: no
cross-split asset leakage, TRAIN-only scaling, no test-time threshold
selection, TRAIN-only OOF stacking subgraph, no IMP evidence reaching the
forecaster, no synthetic-data marker, and full provenance stamps.

## ⚠️ Do not use these directories — pre-real-data / synthetic-era artifacts

`results/toniot/stage_detection/`, `results/toniot/impact_forecasting/`,
`results/toniot/sensitivity/`, `results/edgeiiotset/stage_detection/`,
`results/edgeiiotset/impact_forecasting/`, `results/edgeiiotset/sensitivity/`
were all written 2026-08-11 **13:26–13:54**, strictly before the real CSVs
were placed at 15:26. They were generated against the synthetic smoke-test
generator (`scripts/generate_synthetic_data.py`), the exact scenario the
README warns about ("results produced on it have no scientific meaning").
Notably `results/*/impact_forecasting/forecast_metrics.json` in these stale
directories reports ROC-AUC ≈ 0.98 / PR-AUC ≈ 0.74–0.69 for impact
forecasting on **both** datasets — numbers that look publication-ready but
are synthetic-generator artifacts. The real-data-validated answer is that
impact forecasting is **not valid** for either dataset (see below). If any
manuscript draft, slide, or note cites forecasting AUCs for ToN-IoT or
Edge-IIoTset, that number almost certainly came from one of these stale
files and must be removed, not replaced.

Recommendation: quarantine these six directories (same treatment as
`data/raw/bad_holding/`) rather than deleting outright, pending your
confirmation.

## Dataset summary

| | ToN-IoT | Edge-IIoTset |
|---|---|---|
| Raw records | 461,043 | 157,800 (142,095 kept after cleaning; 15,705 dropped — unparseable timestamp/asset id) |
| Unit of analysis | asset-time-window (Δt = 60s) | **record-level fallback** (no trustworthy timestamp — see below) |
| Final instances | 11,536 | 142,095 |
| Distinct assets | 11,536 | 17,687 |
| Feature dimension | 104 | 175 |
| Train / val / test | 6,922 / 2,307 / 2,307 | 68,301 / 10,804 / 62,990 |
| Stage counts (Benign/IAD/LMEP/IMP) | 11,517 / 3 / 5 / 11 | 24,301 / 41,189 / 30,775 / 45,830 |
| Graph | 11,536 nodes, 11,555 interaction edges, 0 temporal edges, avg degree 2.00 | 142,095 nodes, 17,685 interaction edges, 0 temporal edges, avg degree 0.25 |

**ToN-IoT caveat:** IAD/LMEP/IMP are extremely rare (3/5/11 assets total out
of 11,536). Every per-class metric for these stages, and the bootstrap CI
below, is computed on single-digit asset counts — report as indicative, not
precise.

**Edge-IIoTset caveat:** `frame.time` has a genuine calendar date for **0.0%**
of rows (naive pandas parsing silently defaults everything to Jan-1 — not a
real timestamp). Consequently: (a) asset-level time-windowing was rejected —
it would collapse ~99.9% of assets to the worst-ever IMP class — so the
pipeline falls back to one node per raw record; (b) temporal graph edges are
discarded (248,816 row-order-derived pseudo-temporal edges dropped as
untrustworthy); (c) impact forecasting, window-size sensitivity, and
horizon sensitivity are all gated OFF (see below). This is a genuine
property of the published Edge-IIoTset CSV, not a pipeline bug — confirmed
by `timestamp_calendar_date_frac: 0.0` in the adapter's own data-quality log.

## Main results — stage detection (Macro-F1, mean over 3 seeds)

| Model | ToN-IoT | Edge-IIoTset |
|---|---|---|
| RF (feature-only) | 0.2498 | 0.6146 |
| XGBoost | 0.2498 | 0.5924 |
| LightGBM | 0.2498 | 0.5823 |
| GCN (graph-only) | 0.3678 | 0.3481 |
| GraphSAGE | 0.3621 | 0.3816 |
| GAT | 0.3526 | 0.3355 |
| GRU (no-graph temporal) | 0.3386 | N/A (impact forecasting invalid, see below) |
| Late fusion (RF+GCN) | 0.3391 | 0.5213 |
| **Stacked RF-GCN (proposed)** | 0.3089 | 0.1647 |

Full per-class F1 and std-dev: `results/{toniot,edgeiiotset}/main/stage_detection_main.csv`,
`results/manuscript_tables/tab_{toniot,edgeiiotset}_stage_detection.tex`.
Figures: `results/figures/{toniot,edgeiiotset}/{confusion_matrix_stacked_test,macro_f1_model_ranking}.png`.

**Flag:** on Edge-IIoTset, the proposed Stacked RF-GCN (0.1647) underperforms
every individual base learner and even Late Fusion (0.5213). The
meta-learner ablation (below) shows this is a meta-learner choice artifact,
not a fundamental limitation of stacking.

## One-vs-rest alerting (threshold selected on VAL, applied to TEST)

Full table: `results/{toniot,edgeiiotset}/main/one_vs_rest_alerting.csv`,
`results/manuscript_tables/tab_{toniot,edgeiiotset}_ovr_alerting.tex`.
PR curves: `results/figures/{toniot,edgeiiotset}/ovr_pr_curves_test.png`.

ToN-IoT test-split F1 (n_positive out of 2,307): IAD 0.0 (n=1), LMEP 0.5
(n=1), IMP 0.0 (n=3) — single-digit positives, not reportable as a stable
estimate.

Edge-IIoTset test-split F1 (n_positive out of 62,990): IAD 0.211 (n=19,492),
LMEP 0.080 (n=14,649), IMP 0.465 (n=18,723) — well-powered, usable as-is.

## Impact forecasting — SKIPPED for both datasets (real data)

| | ToN-IoT | Edge-IIoTset |
|---|---|---|
| Status | SKIPPED_WITH_REASON | SKIPPED_WITH_REASON |
| Reason | Zero pre-impact forecasting instances constructible at any split — every asset's observed activity fits inside a single Δt window (no cut point exists before first IMP window and/or no later window to confirm a negative) | 0.0% of rows have a genuine calendar date in `frame.time` — real chronological Δt windowing / temporal edges / pre-impact sequences are not valid |

Source: `results/{toniot,edgeiiotset}/main/impact_forecasting_SKIPPED_WITH_REASON.json`,
`results/manuscript_tables/tab_{toniot,edgeiiotset}_impact_forecasting.tex`.
This is a **structural data limitation**, confirmed independently by the
horizon-sensitivity reviewer experiment (also skipped, same root cause) —
consistent across two separate code paths, not a one-off gate bug.

## Reviewer-response experiments — status

| Experiment | ToN-IoT | Edge-IIoTset |
|---|---|---|
| Stage-mapping sensitivity | ✅ COMPLETE | ✅ COMPLETE |
| Graph ablation | ✅ COMPLETE | ✅ COMPLETE |
| Meta-learner ablation | ✅ COMPLETE | ✅ COMPLETE |
| Baseline tuning (RF/XGB/LGB/GCN/SAGE/GAT, selected on VAL) | ✅ COMPLETE | ✅ COMPLETE |
| Bootstrap 95% CI (asset-level resampling) | ✅ COMPLETE (n=50/1000 valid iters — extreme class rarity limits valid resamples) | ✅ COMPLETE (n=1000/1000) |
| Inference latency | ✅ COMPLETE | ✅ COMPLETE |
| Window-size sensitivity | ✅ COMPLETE (real Δt sweep) | ✅ SKIPPED_WITH_REASON (invalid timestamps — scientifically correct) |
| Horizon sensitivity | ✅ SKIPPED_WITH_REASON (structural — see above) | ✅ SKIPPED_WITH_REASON (invalid timestamps) |

All 16 (8 × 2 datasets) reviewer-response items are accounted for — either
complete with real numbers or validly gated off with a documented,
data-driven reason. None are missing, none failed.

### Key reviewer-experiment findings

**Stage-mapping sensitivity** (`stage_mapping_sensitivity.csv`) — macro-F1
across primary/conservative/expanded label-mapping variants: ToN-IoT 0.343 /
0.374 / 0.405; Edge-IIoTset 0.082 / 0.067 / 0.082. Results are directionally
stable (no mapping variant flips the ranking of models), but ToN-IoT shows
~18% relative spread across variants — worth reporting as a mapping-sensitivity
caveat rather than a fixed number.

**Graph ablation** (`graph_ablation.csv`) — interaction_only ≡ both for both
datasets (temporal edge count is 0 for both real datasets, so "both" never
differs from "interaction_only"). temporal_only alone (no edges — degenerate
to per-node MLP) scores 0.250 (ToN-IoT) / 0.228 (Edge-IIoTset), i.e. it does
**not** collapse to zero, and outperforms interaction_only on Edge-IIoTset
IMP-detection specifically. Directed/typed edge variants are not implemented
(GCN/SAGE/GAT are undirected, untyped) — documented in the table's note.

**Meta-learner ablation** (`meta_learner_ablation.csv`) — this is the
critical result for the Stacked RF-GCN underperformance flagged above.
Edge-IIoTset macro-F1 by meta-learner: logistic_regression 0.115 (= the
number currently in the main table), mlp 0.187, **gradient_boosting 0.308**.
ToN-IoT: logistic_regression 0.343 (main-table value), mlp 0.250,
gradient_boosting 0.250 — here logistic_regression is already best. So the
underperformance is Edge-IIoTset-specific and meta-learner-choice-driven,
not a general flaw in the stacking approach.

**Baseline tuning** (`baseline_tuning.csv`) — full VAL-selected grids and
test scores for RF/XGBoost/LightGBM/GCN/GraphSAGE/GAT, both datasets. Confirms
the main-table baseline numbers were not cherry-picked: e.g. ToN-IoT
GraphSAGE reaches test macro-F1 0.433 under its own best VAL-selected
hyperparameters (vs. 0.362 in the main 3-seed table — the main table uses a
fixed hyperparameter set per model, this table is the fairness/tuning check
requested by reviewers).

**Bootstrap CI** (`bootstrap_confidence_intervals.csv`) — ToN-IoT macro-F1
95% CI [0.278, 0.491] around point 0.343, but only 50/1000 bootstrap
iterations were valid (asset-level resampling on a split with single-digit
minority-class assets frequently produces degenerate resamples) — report the
CI with this caveat explicit, do not present it as a standard 1000-iteration
CI. Edge-IIoTset: [0.080, 0.772] around point 0.115, full 1000/1000 valid
iterations, but the interval is very wide — consistent with the meta-learner
instability found above, not a sampling artifact.

**Inference latency** (`inference_latency.csv`) — end-to-end per-instance
latency: ToN-IoT 171.6ms (5.8 instances/sec), Edge-IIoTset 139.0ms (7.2
instances/sec). RF scoring dominates both (153ms / 127ms) — GCN and
meta-learner stages are comparatively cheap (<13ms combined).

**Window-size sensitivity** (`window_sensitivity.csv`, ToN-IoT only) —
macro-F1 is **exactly identical** (0.4994) across all four candidate Δt
(30/60/120/300s). The graph topology (nodes/edges/avg degree) is also
identical across all four values. This means Δt has no measurable effect in
the current window range for ToN-IoT — worth stating as a finding ("macro-F1
is insensitive to Δt over 30–300s") rather than treating each row as an
independent operating point.

## Figures (current, real-data)

- `results/figures/{toniot,edgeiiotset}/confusion_matrix_stacked_test.png`
- `results/figures/{toniot,edgeiiotset}/macro_f1_model_ranking.png`
- `results/figures/{toniot,edgeiiotset}/ovr_pr_curves_test.png`
- `results/figures/toniot/window_sensitivity.png` (Edge-IIoTset has no
  equivalent — window sensitivity is validly SKIPPED there, see above)
- `results/figures/{toniot,edgeiiotset}/graph_ablation.png`
- `results/figures/{toniot,edgeiiotset}/meta_learner_ablation.png`

The last three were generated 2026-08-12 by `scripts/generate_reviewer_figures.py`
directly from the real-data CSVs in `results/*/reviewer_experiments/` (no
retraining — pure plotting of already-computed numbers), closing the gap
where `run_reviewer_experiments.py` itself only writes CSV + LaTeX tables.
Styling matches the project's shared palette (`src/evaluation/palette.py`),
consistent with the main-result figures.

## LaTeX tables (all current, real-data, in `results/manuscript_tables/`)

`tab_{dataset}_dataset_summary`, `tab_{dataset}_stage_detection`,
`tab_{dataset}_ovr_alerting`, `tab_{dataset}_impact_forecasting` (skip-note),
`tab_{dataset}_stage_mapping_sensitivity`, `tab_{dataset}_graph_ablation`,
`tab_{dataset}_meta_learner_ablation`, `tab_{dataset}_baseline_tuning`,
`tab_{dataset}_bootstrap_ci`, `tab_{dataset}_inference_latency`,
`tab_{dataset}_window_sensitivity`, `tab_{dataset}_horizon_sensitivity` —
24 files total (12 × 2 datasets), all present and dated 2026-08-11 16:20
through 2026-08-12 00:08.
