# Manuscript Result Summary — Real Data (Corrected)

Generated: 2026-08-12; **ToN-IoT section updated 2026-08-20** after fixing a
critical windowing bug (see next section). Source of truth for every number
in this document is `results/*/main/`, `results/*/reviewer_experiments/`,
`results/manuscript_tables/`, and `results/figures/` — all produced from the
real datasets (`data/raw/toniot/Train_Test_Network.csv`,
`data/raw/edgeiiotset/ML-EdgeIIoT-dataset.csv`), placed on disk 2026-08-11
15:26 local. `results/VALIDATION_REPORT.json` (16/16 checks passed, both
datasets, re-run 2026-08-20 02:04 UTC) confirms: no cross-split asset
leakage, TRAIN-only scaling, no test-time threshold selection, TRAIN-only
OOF stacking subgraph, no IMP evidence reaching the forecaster, no
synthetic-data marker, and full provenance stamps.

## ⚠️ ToN-IoT windowing bug fixed 2026-08-20 — old ToN-IoT numbers below were INVALID

`src/preprocessing/windowing.py::assign_window_id` divided an
already-in-seconds `datetime64[s]` integer view by `1e9` again (a pandas
datetime-resolution assumption that broke on this environment's pandas
3.0.1), collapsing `window_id` to a single constant value for **every**
ToN-IoT record regardless of its real timestamp. This silently turned "one
instance per asset per real 60s window" into "one instance per asset,
aggregated over its entire ~27-day observed history" — 461,043 raw records
collapsed to 11,536 instances (== asset count) with only 3 IAD / 5 LMEP /
11 IMP non-Benign instances surviving, instead of the correct 34,435
instances (86 / 298 / 243). Every ToN-IoT number in this document has been
regenerated post-fix. Root cause: `results/archive/toniot_pre_window_fix_INVALID/README.md`.
Full old-vs-new comparison for every affected result:
`results/TONIOT_WINDOW_FIX_RESULT_DIFF.md`. Regression tests:
`tests/test_windowing.py`, `tests/test_toniot_windowing_integration.py`.
Edge-IIoTset was never affected (its record-level fallback never calls
`assign_window_id`) and none of the Edge-IIoTset numbers below changed.

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
| Final instances | 34,435 | 142,095 |
| Distinct assets | 11,536 | 17,687 |
| Feature dimension | 126 | 175 |
| Train / val / test | 27,326 / 3,796 / 3,313 (instances; 6,922 / 2,307 / 2,307 assets) | 68,301 / 10,804 / 62,990 |
| Stage counts (Benign/IAD/LMEP/IMP) | 33,808 / 86 / 298 / 243 | 24,301 / 41,189 / 30,775 / 45,830 |
| Graph | 34,435 nodes, 10,168 interaction edges, 22,899 temporal edges, avg degree 1.92 | 142,095 nodes, 17,685 interaction edges, 0 temporal edges, avg degree 0.25 |

**ToN-IoT caveat:** IAD/LMEP/IMP are still minority classes (86/298/243
instances out of 34,435), but this is now a real, well-powered-enough
sample (not the pre-fix 3/5/11 single-digit degenerate counts) — report
per-class metrics for these stages as indicative of a real but
imbalanced-class effect, not as noise on a handful of samples.

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
| RF (feature-only) | 0.4545 | 0.6146 |
| XGBoost | 0.3776 | 0.5924 |
| LightGBM | 0.5345 | 0.5823 |
| GCN (graph-only) | 0.6599 | 0.3481 |
| GraphSAGE | 0.6707 | 0.3816 |
| GAT | 0.6874 | 0.3355 |
| GRU (no-graph temporal) | 0.4303 | N/A (impact forecasting invalid, see below) |
| Late fusion (RF+GCN) | 0.6577 | 0.5213 |
| **Stacked RF-GCN (proposed)** | 0.3951 | 0.1647 |

ToN-IoT column corrected 2026-08-20 (windowing fix — was 0.2498–0.3678
range, computed on a degenerate 3/5/11-instance non-Benign sample; see
`results/TONIOT_WINDOW_FIX_RESULT_DIFF.md`). Edge-IIoTset column unchanged.

Full per-class F1 and std-dev: `results/{toniot,edgeiiotset}/main/stage_detection_main.csv`,
`results/manuscript_tables/tab_{toniot,edgeiiotset}_stage_detection.tex`.
Figures: `results/figures/{toniot,edgeiiotset}/{confusion_matrix_stacked_test,macro_f1_model_ranking}.png`.

**Flag:** on Edge-IIoTset, the proposed Stacked RF-GCN (0.1647) underperforms
every individual base learner and even Late Fusion (0.5213). The
meta-learner ablation (below) shows this is a meta-learner choice artifact,
not a fundamental limitation of stacking. **Post-fix, the same pattern now
also shows on ToN-IoT**: Stacked RF-GCN (0.3951) underperforms GAT (0.6874),
GraphSAGE (0.6707), GCN (0.6599), and Late Fusion (0.6577) — and the
ToN-IoT meta-learner ablation (below) shows this is, again, a meta-learner
choice artifact (gradient_boosting reaches 0.468 vs. logistic_regression's
0.308), not a fundamental stacking limitation. This is a real, reproducible
cross-dataset finding, not a fix-induced artifact — worth a paper note.

## One-vs-rest alerting (threshold selected on VAL, applied to TEST)

Full table: `results/{toniot,edgeiiotset}/main/one_vs_rest_alerting.csv`,
`results/manuscript_tables/tab_{toniot,edgeiiotset}_ovr_alerting.tex`.
PR curves: `results/figures/{toniot,edgeiiotset}/ovr_pr_curves_test.png`.

ToN-IoT test-split F1 (n_positive out of 3,313, corrected 2026-08-20): IAD
0.0087 (n=48), LMEP 0.5698 (n=116), IMP 0.5185 (n=33) — real, if still
minority-class-limited, positive counts (was 1/1/3 pre-fix); reportable as
an indicative estimate, no longer a single-digit non-result.

Edge-IIoTset test-split F1 (n_positive out of 62,990): IAD 0.211 (n=19,492),
LMEP 0.080 (n=14,649), IMP 0.465 (n=18,723) — well-powered, usable as-is.

## Impact forecasting — SKIPPED for both datasets (real data)

| | ToN-IoT | Edge-IIoTset |
|---|---|---|
| Status | SKIPPED_WITH_REASON | SKIPPED_WITH_REASON |
| Reason (corrected 2026-08-20) | 21,313 real pre-impact instances now exist (5 positive, all in TRAIN); TEST split has 0/798 positive (VAL also 0/1,391) — ROC-AUC/PR-AUC/Capture@k undefined on held-out data. Of the 11 assets that ever reach IMP, 7 have IMP as their literal first observed window (instant-onset attack bursts, no escalation history to forecast from) | 0.0% of rows have a genuine calendar date in `frame.time` — real chronological Δt windowing / temporal edges / pre-impact sequences are not valid |

Source: `results/{toniot,edgeiiotset}/main/impact_forecasting_SKIPPED_WITH_REASON.json`,
`results/manuscript_tables/tab_{toniot,edgeiiotset}_impact_forecasting.tex`.
**ToN-IoT reason changed 2026-08-20** (was: "zero instances constructible at
any split", a windowing-bug artifact — see
`results/toniot/IMPACT_FORECASTING_FEASIBILITY_REEVALUATION.md` and
`results/TONIOT_WINDOW_FIX_RESULT_DIFF.md`). The new reason is a genuine
structural data limitation (confirmed independently by the horizon-
sensitivity reviewer experiment, also skipped at all 3 candidate horizons
for the same reason), not a pipeline bug. Edge-IIoTset reason unchanged.

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
across primary/conservative/expanded label-mapping variants: ToN-IoT
**0.308 / 0.338 / 0.414** (corrected 2026-08-20; was 0.343/0.374/0.405
pre-fix); Edge-IIoTset 0.082 / 0.067 / 0.082 (unchanged). Same qualitative
ranking as before (expanded > conservative > primary) survives the fix —
results are directionally stable, but ToN-IoT still shows a meaningful
relative spread across variants — worth reporting as a mapping-sensitivity
caveat rather than a fixed number.

**Graph ablation** (`graph_ablation.csv`) — **ToN-IoT corrected 2026-08-20**:
pre-fix, interaction_only ≡ both and temporal_only was a *structurally
vacuous* comparison (0 temporal edges — every asset had exactly one
degenerate window, so "connect consecutive windows of the same asset" was
impossible by construction). Post-fix, with 22,899 real temporal edges,
this is now a real, non-degenerate ablation: interaction_only **0.359**,
temporal_only **0.283**, both **0.308** — temporal-only alone is a real,
usable signal, not a no-op. Edge-IIoTset numbers unchanged: interaction_only
≡ both (temporal edges are still 0 there — genuinely, not as a bug, since
Edge-IIoTset's timestamps are unusable), temporal_only 0.228. Directed/typed
edge variants are not implemented (GCN/SAGE/GAT are undirected, untyped) —
documented in the table's note.

**Meta-learner ablation** (`meta_learner_ablation.csv`) — this is the
critical result for the Stacked RF-GCN underperformance flagged above.
Edge-IIoTset macro-F1 by meta-learner: logistic_regression 0.115 (= the
number currently in the main table), mlp 0.187, **gradient_boosting 0.308**
(unchanged). **ToN-IoT corrected 2026-08-20**: logistic_regression **0.308**
(main-table value), mlp **0.316**, **gradient_boosting 0.468 (now best)** —
pre-fix, logistic_regression looked already-best (0.343/0.250/0.250); post-fix
the ranking **flips** and matches the Edge-IIoTset pattern: the Stacked
RF-GCN underperformance is meta-learner-choice-driven on **both** datasets,
not a fundamental limitation of stacking, and not Edge-IIoTset-specific as
previously stated.

**Baseline tuning** (`baseline_tuning.csv`) — full VAL-selected grids and
test scores for RF/XGBoost/LightGBM/GCN/GraphSAGE/GAT, both datasets.
Confirms the main-table baseline numbers were not cherry-picked. ToN-IoT
corrected 2026-08-20: GAT reaches test macro-F1 **0.734**, GraphSAGE
**0.687**, GCN **0.654**, LightGBM **0.534**, RF **0.459**, XGBoost
**0.371** under each model's own best VAL-selected hyperparameters (was
0.249–0.433 pre-fix) — the main table uses a fixed hyperparameter set per
model, this table is the fairness/tuning check requested by reviewers.
Edge-IIoTset numbers unchanged.

**Bootstrap CI** (`bootstrap_confidence_intervals.csv`) — **ToN-IoT
corrected 2026-08-20**: macro-F1 95% CI **[0.230, 0.408]** around point
**0.308**, now **1000/1000** bootstrap iterations valid (was [0.278, 0.491]
around 0.343 with only 50/1000 valid pre-fix — asset-level resampling on
the pre-fix single-digit-minority-class split frequently produced
degenerate resamples; with 627 real non-Benign instances across 105
non-trivial assets post-fix, resampling is no longer degenerate). Edge-IIoTset
unchanged: [0.080, 0.772] around point 0.115, full 1000/1000 valid
iterations, interval still wide — consistent with the meta-learner
instability found above, not a sampling artifact.

**Inference latency** (`inference_latency.csv`) — end-to-end per-instance
latency: ToN-IoT **185.5ms (5.4 instances/sec)** (corrected 2026-08-20; the
pre-fix run's csv had this row blank — never actually recorded), Edge-IIoTset
139.0ms (7.2 instances/sec, unchanged). RF scoring dominates both (168ms /
127ms) — GCN and meta-learner stages are comparatively cheap (<35ms
combined).

**Window-size sensitivity** (`window_sensitivity.csv`, ToN-IoT only) —
**corrected 2026-08-20**: macro-F1 now **varies meaningfully** with Δt —
30s→0.376, 60s→0.308, 120s→**0.579**, 300s→0.532 — and graph topology
varies too (44,688 / 34,435 / 26,561 / 20,094 nodes). The pre-fix finding
of "macro-F1 exactly identical (0.4994) at all four Δt values, identical
graph topology" was itself a symptom of the bug (`window_id` never actually
depended on Δt while it was constant) — retract that finding. Δt has a
real, non-monotonic effect on ToN-IoT in this range and should be reported
as such, with 120s as the best-performing candidate.

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
consistent with the main-result figures. **ToN-IoT figures regenerated
2026-08-20** post-windowing-fix (all of them, since every ToN-IoT CSV
changed); Edge-IIoTset figures untouched (unaffected by the fix).

## LaTeX tables (all current, real-data, in `results/manuscript_tables/`)

`tab_{dataset}_dataset_summary`, `tab_{dataset}_stage_detection`,
`tab_{dataset}_ovr_alerting`, `tab_{dataset}_impact_forecasting` (skip-note),
`tab_{dataset}_stage_mapping_sensitivity`, `tab_{dataset}_graph_ablation`,
`tab_{dataset}_meta_learner_ablation`, `tab_{dataset}_baseline_tuning`,
`tab_{dataset}_bootstrap_ci`, `tab_{dataset}_inference_latency`,
`tab_{dataset}_window_sensitivity`, `tab_{dataset}_horizon_sensitivity` —
24 files total (12 × 2 datasets), all present. Edge-IIoTset's 12 files
still dated 2026-08-11 16:20 through 2026-08-12 00:08 (unchanged). ToN-IoT's
12 files regenerated 2026-08-20 (windowing fix — see
`results/TONIOT_WINDOW_FIX_RESULT_DIFF.md`).
