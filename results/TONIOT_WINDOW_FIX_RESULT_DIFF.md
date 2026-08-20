# ToN-IoT Windowing-Bug Fix — Result Diff (Step 7)

**All rows below marked "OLD (INVALID)" were produced while `window_id` was
constant across the entire dataset** (`src/preprocessing/windowing.py::assign_window_id`
divided an already-in-seconds `datetime64[s]` integer view by `1e9` again,
crushing every ToN-IoT timestamp to `floor(x/60) == 0`). Every OLD number
below is therefore not "every asset aggregated over a real 60s window" but
"every asset aggregated over its **entire ~27-day observed history**,
collapsed to one instance." See `results/archive/toniot_pre_window_fix_INVALID/README.md`
for the root-cause trace and `results/toniot/WINDOWING_FIX_AUDIT.md` for the
post-fix audit. **The OLD numbers must not be cited, quoted, or restored —
they are archived only for this diff.**

NEW numbers: `results/toniot/main/*`, `results/toniot/reviewer_experiments/*`,
`results/manuscript_tables/tab_toniot_*.tex`, generated 2026-08-20 by
`scripts/run_main_results.py --dataset toniot --seeds 42 43 44` and
`scripts/run_reviewer_experiments.py --dataset toniot --seed 42`, against
the SAME raw file (`data/raw/toniot/Train_Test_Network.csv`, unchanged) —
the only code change between OLD and NEW is the `assign_window_id` fix.
`results/VALIDATION_REPORT.json` re-confirms 16/16 structural checks (both
datasets) after the fix.

## Dataset summary / class support

| | OLD (INVALID) | NEW (corrected) | Reason |
|---|---|---|---|
| Unit of analysis | 1 instance = 1 asset's **entire lifetime** (window_id constant) | 1 instance = 1 asset **x real 60s window** | window_id now varies (4,100 distinct global buckets) |
| Final instances | 11,536 | **34,435** | equals `n_distinct_assets` pre-fix (degenerate); now `n_distinct_assets` (11,536) x (avg 2.985 windows/asset) |
| Feature dimension | 104 | 126 | 22 more numeric/categorical columns now carry within-asset variance across real windows (std/max/nunique aggregates were previously computed over a single degenerate group per asset) |
| Train / val / test | 6,922 / 2,307 / 2,307 (asset counts) | 27,326 / 3,796 / 3,313 (window-instance counts; same 6,922/2,307/2,307 **assets**, asset-disjoint split unchanged) | more instances per asset, same split assignment (same seed=42, same asset partition) |
| Stage counts (Benign/IAD/LMEP/IMP) | 11,517 / **3** / **5** / **11** | 33,808 / **86** / **298** / **243** | `max_severity` was collapsing each asset's ENTIRE history to its single worst-ever stage; now computed per real window, so an asset with both benign and attack history contributes separate windows for each, not one obliterated label |
| Graph nodes | 11,536 | 34,435 | = new instance count |
| Graph interaction edges (undirected) | 11,555 | 10,168 | fewer, because "same window" now means a real 60s co-occurrence, not "ever communicated at any point in 27 days" |
| Graph temporal edges (undirected) | **0** | **22,899** | structurally impossible pre-fix — temporal edges connect an asset's *consecutive* windows, and every asset had exactly 1 (degenerate) window |
| Graph avg degree | 2.0033 | 1.9205 | recomputed from the above |

**This is the single most direct confirmation of the bug**: temporal edges
went from a hard-coded 0 (not "rare" — *structurally impossible* under
constant `window_id`) to 22,899, because real Δt-windowing is the only
thing that lets an asset have more than one graph node to connect in time.

## Stage-detection model comparison (macro-F1, mean over seeds 42/43/44)

| Model | OLD (INVALID) | NEW (corrected) | Δ |
|---|---|---|---|
| RF (feature-only) | 0.2498 | **0.4545** | +0.205 |
| XGBoost | 0.2498 | **0.3776** | +0.128 |
| LightGBM | 0.2498 | **0.5345** | +0.285 |
| GCN (graph-only) | 0.3678 | **0.6599** | +0.292 |
| GraphSAGE | 0.3621 | **0.6707** | +0.309 |
| GAT | 0.3526 | **0.6874** | +0.335 |
| GRU (no-graph temporal) | 0.3386 | **0.4303** | +0.092 |
| Late fusion (RF+GCN) | 0.3391 | **0.6577** | +0.319 |
| Stacked RF-GCN (proposed) | 0.3089 | **0.3951** | +0.086 |

Reason for every row: OLD macro-F1 was computed on a test split with a
literal 3/5/11 (IAD/LMEP/IMP) surviving instances **total across the whole
dataset** — the near-constant ~0.25–0.37 spread across models was mostly
measuring noise on single-digit minority classes, not real discriminative
skill. NEW macro-F1 is computed on the real class distribution (627
non-Benign windows total, properly spread across train/val/test) and shows
a much more informative ranking. **This ranking is a real (if still
minority-class-limited) result, not yet a "final" one** — GAT/GraphSAGE/GCN
clearly outperform the proposed Stacked RF-GCN, which is a genuine finding
that survived the fix, not an artifact of it (see Meta-learner ablation
below for why).

## One-vs-rest alerting (test split)

| Stage | OLD F1 (n_positive/2,307) | NEW F1 (n_positive/3,313) | Reason |
|---|---|---|---|
| IAD | 0.0 (n=1) | 0.0087 (n=48) | still poor (IAD is intrinsically the rarest/least separable class), but n went from 1 to 48 — no longer a single-digit-sample non-result |
| LMEP | 0.5 (n=1) | 0.5698 (n=116) | n went from 1 to 116; NEW is a real, reportable number |
| IMP | 0.0 (n=3) | 0.5185 (n=33) | n went from 3 to 33; OLD 0.0 was an artifact of near-zero test-split positives, not a real model failure |

## Impact forecasting

| | OLD (INVALID) | NEW (corrected) |
|---|---|---|
| Status | SKIPPED_WITH_REASON | **still SKIPPED_WITH_REASON — different reason** |
| Instances constructed | 0 | 21,313 (5 positive, 21,308 negative) |
| Reason | "zero pre-impact forecasting instances could be constructed at any split — every asset's observed activity fits inside a single Delta-t window" | "TEST split has only one class present (0 positive / 798 negative of 798 instances) — ROC-AUC/PR-AUC/Capture@k are undefined without both classes." (VAL also has 0/1,391 positive, same problem, not currently asserted by the gate but confirmed in `results/toniot/IMPACT_FORECASTING_FEASIBILITY_REEVALUATION.md`) |
| Root cause | windowing bug (this fix) | genuine dataset property: of 11 assets that ever reach IMP, 7 have IMP as their literal first observed window (instant-onset attack bursts, no pre-impact history); of the 4 with real pre-impact history, only 2 (both landing in TRAIN under the asset-disjoint split) have any window inside the 30-minute horizon |
| **Do not train/report a GRU forecaster for ToN-IoT** — this is now a real, well-characterized negative result, not a pipeline bug. | | |

Horizon sensitivity (H=10/30/60): OLD skipped for the same (bug) reason as
the main forecaster; NEW skipped independently at all three horizons for
the same (real, data-driven) reason — consistent across both code paths,
same as OLD, but now for the correct underlying cause.

## Reviewer-response experiments

| Experiment | OLD (INVALID) | NEW (corrected) |
|---|---|---|
| Stage-mapping sensitivity (macro-F1) | primary 0.343 / conservative 0.374 / expanded 0.405 | primary **0.308** / conservative **0.338** / expanded **0.414** — same qualitative ranking (expanded > conservative > primary), NEW primary is slightly lower because the real (much larger) LMEP/IMP class populations are harder than the single-digit OLD samples suggested |
| Graph ablation | interaction_only ≡ both (0.343); temporal_only 0.250, **0 temporal edges** (degenerate — literally could not differ from a no-edge MLP) | interaction_only **0.359**, temporal_only **0.283** (now backed by 22,899 real temporal edges, no longer a structural no-op), both **0.308** — ablation is now a real comparison, not a vacuous one |
| Meta-learner ablation (ToN-IoT) | logistic_regression 0.343 (best) / mlp 0.250 / gradient_boosting 0.250 | logistic_regression 0.308 / mlp 0.316 / **gradient_boosting 0.468 (best)** — the ranking **flipped**: gradient_boosting is now clearly best, matching the Edge-IIoTset finding already in the manuscript (the Stacked RF-GCN main-table result uses logistic_regression; switching to gradient_boosting would materially close the gap to GAT/GraphSAGE/GCN — worth a manuscript note) |
| Baseline tuning (VAL-selected, test macro-F1) | RF 0.2498, XGBoost 0.2498, LightGBM 0.2498, GCN 0.3208, GraphSAGE 0.4330, GAT 0.3329 | RF **0.4586**, XGBoost **0.3705**, LightGBM **0.5340**, GCN **0.6540**, GraphSAGE **0.6868**, GAT **0.7343** — same conclusion as the main table (GNNs > tree baselines), now on real class support |
| Bootstrap 95% CI (macro-F1) | point 0.343, CI [0.278, 0.491], **50/1000 valid iterations** | point 0.308, CI [0.230, 0.408], **1000/1000 valid iterations** | the OLD CI's own caveat ("only 50/1000 valid — extreme class rarity") is resolved: asset-level resampling on 627 real non-Benign windows across 105 non-trivial assets no longer produces degenerate resamples |
| Inference latency (end-to-end) | not computed (field blank in OLD csv) | 185.5ms / instance, 5.4 instances/sec (RF scoring 167.8ms dominates) | OLD run apparently didn't reach/record this row; NEW is complete |
| Window-size sensitivity | macro-F1 **exactly identical** (0.4994) at all 4 Δt values (30/60/120/300s); graph topology also identical at all 4 | macro-F1 **varies** by Δt: 30s→0.376, 60s→0.308, 120s→**0.579**, 300s→0.532; graph topology now varies with Δt (44,688 / 34,435 / 26,561 / 20,094 nodes) | OLD "no sensitivity to Δt" was itself a bug artifact (Δt never actually affected `window_id` since it was always floor(~1.55/Δt)=0 for any Δt in this candidate range); Δt now has a real, non-monotonic effect worth reporting as a genuine finding, not "insensitive" |

## What did NOT change (as expected)

- Edge-IIoTset: untouched. `prepare_dataset('edgeiiotset')` still returns
  142,095 record-level instances via the `build_record_level_instances`
  fallback, which never calls `assign_window_id`. `run_final_validation.py
  --datasets toniot edgeiiotset` re-confirms 16/16 structural checks
  passing for both datasets (re-run only to restore the shared
  `VALIDATION_REPORT.json` after an inadvertent overwrite — no Edge-IIoTset
  model was retrained beyond this validation script's own internal
  structural checks, and its instance counts/graph topology are identical
  to before).
- Asset-disjoint split: same seed (42), same asset→split assignment
  (6,922/2,307/2,307 assets) — only the number of *instances per asset*
  changed, not which asset is in which split.
- Leakage properties: 8/8 (toniot) structural checks still pass — no
  cross-split asset leakage, TRAIN-only scaling, TRAIN-only OOF stacking
  subgraph, no IMP evidence reaching the forecaster.

## Recommendation for the manuscript

1. Replace every ToN-IoT number sourced from the pre-fix `main/` and
   `reviewer_experiments/` directories (now archived under
   `results/archive/toniot_pre_window_fix_INVALID/`) with the corrected
   numbers above.
2. Report Stage detection with the NEW ranking (GAT > GraphSAGE > GCN >
   Late fusion > LightGBM > GRU > RF > Stacked RF-GCN > XGBoost) — flag
   that the proposed Stacked RF-GCN underperforming individual GNN base
   learners is a genuine, reproducible finding (also true pre-fix, in a
   noisier form), and that switching its meta-learner to gradient_boosting
   (0.468 vs. 0.308 macro-F1) is a concrete, evidenced improvement worth
   adopting or at least discussing.
3. Report impact forecasting as SKIPPED with the NEW reason (real
   pre-impact sequences exist, but ToN-IoT's attack episodes are
   overwhelmingly instant-onset with no observable escalation history, and
   the handful of exceptions all fall in TRAIN under the asset-disjoint
   split) — this is a materially different, more defensible scientific
   statement than the OLD "zero instances" reason and should replace it
   verbatim, not be merged with it.
4. Report window-size sensitivity as a real (non-monotonic) effect, not
   "insensitive to Δt."
