# Manuscript vs. Validated Real-Data Results — Audit

Compares `manuscript/current_manuscript.tex` (1209 lines, authors: Afrifah,
Epiphaniou, Lallie, Maple) against `results/MANUSCRIPT_RESULT_SUMMARY.md`
and the underlying validated CSV/JSON files under `results/`
(`VALIDATION_REPORT.json`: 16/16 checks passed, both datasets, real data).
**No edits have been made to `current_manuscript.tex`.** This document is
the audit only.

No response-to-reviewers letter exists anywhere in this repo or the
surrounding filesystem — only the manuscript `.tex` was provided. The
manuscript's own `\color{red}` markup identifies exactly one traceable
reviewer-driven edit (the Edge-IIoTset external-validation addition); every
other reviewer-comment mapping in §8 below is **inferred** from what the
reviewer-experiment suite conventionally answers, not confirmed against an
actual review. Supply the letter if you want firm R1/R2 attribution.

---

## 0. Headline finding — the manuscript describes an experiment that was never run

This dominates almost every other finding below, so it's stated first.

The manuscript's Experiments section (Table `raw_modality_splits`, Table
`summary_stats_from_logs`, Table `graph_hparams_stats`, and the prose in
§5.1) describes a **multi-modal** ToN-IoT pipeline fusing IoT device
telemetry (7 streams), network flows, and Linux/Windows host logs into
**N = 759,582 unified instances** with **F = 310 features**, on a graph of
**759,582 nodes / 3,036,716 edges**.

The validated, provenance-stamped, leakage-checked pipeline that actually
produced every number in `results/` uses **only** the ToN-IoT network-flow
CSV:

- `configs/toniot.yaml` header: *"Dataset configuration: ToN-IoT (Network
  flow subset)... Train_Test_Network.csv, ~461,043 rows, 44 columns"*.
- `src/data/` contains exactly two adapters: `toniot_adapter.py` and
  `edgeiiotset_adapter.py` — both single-CSV, network-flow-schema parsers.
  There is no IoT-telemetry adapter, no Linux-log adapter, no Windows-log
  adapter anywhere in the codebase.
- `data/raw/toniot/` has only ever contained `Train_Test_Network.csv` (and
  its download-staging copy) — never any IoT/Linux/Windows files.
- Real, validated result: `raw_records_count = 461,043`,
  `final_unified_instances = 11,536`, `feature_dimension = 104`,
  `graph_nodes = 11,536`, `graph_interaction_edges = 11,555`.

**11,536 instances / 104 features is what the codebase, as it exists and as
validated, actually produces — not 759,582 / 310.** The multi-modal fusion
architecture described in §4.2 (feature construction mentioning fridge/
thermostat/GPS telemetry and Linux/Windows counters) is a genuine design
capability of the *conceptual* framework, but it was never implemented or
executed. Every downstream number that traces back to the 759,582-instance
scale (dataset tables, the abstract's headline macro-F1, the OvR table, the
seed-robustness table, the scalability paragraph) is therefore **not
supported by any file in `results/`** — not even the stale synthetic-era
ones, which are also single-modality. This is a scope gap, not a rounding
error: two options going forward are (a) actually build and run the
multi-modal pipeline before publishing these numbers, or (b) rescope the
paper honestly to the network-flow-only evaluation that was actually run,
moving multi-modal fusion to Future Work. Given the timeline implied by
this session, (b) is what the LaTeX below assumes; swap in real multi-modal
numbers instead if (a) becomes available before submission.

---

## 1. Stage-detection results — real numbers reverse the paper's central claim

| Model | Manuscript (Table `tab:main_4class_results_all`) | Real ToN-IoT (validated, 3-seed mean) | Real Edge-IIoTset (validated, 3-seed mean) |
|---|---|---|---|
| RF (feature-only) | 0.786 ± 0.006 | **0.2498 ± 0.000** | **0.6146 ± 0.003** |
| GCN (graph-only) | 0.793 ± 0.007 | **0.3678 ± 0.047** | **0.3481 ± 0.136** |
| Late fusion | 0.806 ± 0.005 | **0.3391 ± 0.064** | **0.5213 ± 0.047** |
| **Stacked RF-GCN (proposed)** | **0.816 ± 0.004 (best)** | **0.3089 ± 0.047** | **0.1647 ± 0.060 (worst)** |
| XGBoost/LightGBM | 0.801 ± 0.006 | XGB 0.2498 / LGBM 0.2498 | XGB 0.5924 / LGBM 0.5823 |
| GRU (no-graph) | 0.795 ± 0.007 | 0.3386 ± 0.008 | N/A — impact-forecasting-invalid dataset, no temporal baseline applicable |
| GraphSAGE | 0.800 ± 0.007 | 0.3621 ± 0.050 | 0.3816 ± 0.058 |
| GAT | 0.804 ± 0.006 | 0.3526 ± 0.074 | 0.3355 ± 0.141 |

Two independent real datasets both show **the proposed Stacked RF-GCN is
not the best model** — it underperforms GCN-alone and Late Fusion on
ToN-IoT, and underperforms *every other model including plain RF* on
Edge-IIoTset. This is the single most consequential finding of the audit:
the manuscript's central architectural claim ("stacking achieves the
highest Macro-F1... indicating complementary signals") is **not supported**
on real data for either dataset, and the abstract's 0.8159 headline number
is not achievable with the current implementation and this scale of data.

**Root cause, from the meta-learner ablation (real, both datasets):**
Edge-IIoTset macro-F1 by meta-learner: logistic_regression 0.115 (≈ what
the main table reports, 0.1647 avg), mlp 0.187, **gradient_boosting
0.308**. ToN-IoT: logistic_regression 0.343 (best there — matches main),
mlp 0.250, gradient_boosting 0.250. So the underperformance is
Edge-IIoTset-specific and driven by the meta-learner choice (logistic
regression), not a fundamental flaw in stacking as a strategy — this is a
genuine, defensible mitigation finding to report, but it does not rescue
the *as-configured, as-reported* main result.

**Action:** replace Table `tab:main_4class_results_all` with real numbers
(both datasets — LaTeX below), rewrite every paragraph asserting stacking
superiority, and reframe the contribution honestly: stacking is proposed
and its leakage-safe protocol is implemented and validated, but on the two
real datasets evaluated, it is not the top performer as configured; the
meta-learner ablation identifies a concrete, testable fix (gradient
boosting meta-learner) that is left as the recommended configuration change
rather than a claimed result (its own single-seed number, not a 3-seed
mean, so don't promote it as the new headline either).

---

## 2. Impact forecasting — invalid on both real datasets, not merely "different numbers"

This is not a numbers substitution — the task is **structurally impossible**
on both real datasets as currently defined, confirmed by two independent
code paths (the main pipeline's own gate, and the reviewer horizon-
sensitivity sweep, which fails identically):

- **ToN-IoT:** *"Zero pre-impact forecasting instances could be constructed
  at any split — every asset's observed activity fits inside a single Δt
  window (no earlier 'cut point' exists before the asset's first IMP window,
  and/or no later window exists to confirm a negative)."* Tested at H =
  10, 30, 60 — fails at all three.
- **Edge-IIoTset:** `frame.time` has a genuine calendar date for **0.0%**
  of rows (naive parsing silently defaults everything to Jan-1) — no
  chronological ordering exists to forecast across.

Every forecasting number in the manuscript is therefore unsupported:

| Manuscript claim | Location | Status |
|---|---|---|
| Abstract: "ROC-AUC 0.9047 and PR-AUC 0.4186... captures 0.6820... top 5%" | Abstract, last sentence | **Remove** |
| Table `tab:impact_split_counts` (Train 165/7, Val 35/1, Test 36/2) | §6.2, `\label{tab:impact_split_counts}` | Manuscript itself flags this as *"illustrative values... should be replaced by the actual counts used in the final experiments"* — never was. **Remove**, replace with SKIPPED note. |
| Table `tab:impact_results_combined` (Stage-only/Embedding-only/Dual-stream GRU, Capture@1/2/5/10%) | §6.2 | **Remove** |
| `tab:seed_robustness` row "Impact prediction — Dual-stream GRU — PR-AUC: 0.419±0.018" | §6.3 | **Remove** |
| "10/236 positives... 16/243 positives, as observed in experimental logs" | §6.2, imbalance paragraph | Manuscript's own words flag this as illustrative/log-derived, not a controlled experiment. **Remove.** |
| Entire "Held-out performance and triage ranking" / "Where the dual-stream design helps" / "Failure cases" prose | §6.2 | Built entirely on the removed tables. **Rewrite** as an honest negative-result section (below). |

**What survives:** the *leakage-control engineering* claims are true and
independently verified by `VALIDATION_REPORT.json` (`forecaster_pre_cutoff_only`,
`no_imp_evidence_in_forecaster_features` — both PASS, both datasets, real
data). So "the protocol never lets IMP evidence reach the forecaster" is a
fully supported claim; "the forecaster achieves ROC-AUC 0.90" is not. These
are different claims and must be separated in the rewrite — the paper can
honestly say it built and verified a leakage-safe forecasting protocol, and
that it correctly *refuses* to report fabricated numbers when the data
can't support the task, which is itself a defensible methodological
contribution (most papers wouldn't catch this).

---

## 3. One-vs-rest alerting — ToN-IoT numbers are single-digit-count noise; Edge-IIoTset is well-powered and new

| Stage | Manuscript (ToN-IoT, claimed) | Real ToN-IoT (test, n=2307) | Real Edge-IIoTset (test, n=62,990) |
|---|---|---|---|
| IAD | P 0.393 R 0.539 F1 0.454 PR-AUC 0.312 | P 0 R 0 **F1 0** PR-AUC 0.053 (n_pos=**1**) | P 0.569 R 0.129 F1 0.211 PR-AUC 0.351 (n_pos=19,492) |
| LMEP | P 0.827 R 0.538 F1 0.652 PR-AUC 0.701 | P 0.333 R 1.0 **F1 0.5** PR-AUC 0.333 (n_pos=**1**) | P 0.078 R 0.081 F1 0.080 PR-AUC 0.157 (n_pos=14,649) |
| IMP | P 0.9998 R 0.154 F1 0.267 PR-AUC 0.585 | P 0 R 0 **F1 0** PR-AUC 0.340 (n_pos=**3**) | P 0.727 R 0.342 F1 0.465 PR-AUC 0.503 (n_pos=18,723) |

ToN-IoT's test split has 1, 1, and 3 positive assets for IAD/LMEP/IMP out
of 2,307 — any P/R/F1 computed there is a single-prediction coin-flip, not
an operating point. **Action:** report the real ToN-IoT numbers but with an
explicit "not a stable estimate" caveat (don't present F1=0 or F1=0.5 as if
they were meaningful precision/recall trade-offs), and **promote
Edge-IIoTset's OvR table as the primary, well-powered one-vs-rest result** —
it didn't exist in the manuscript at all before this audit.

---

## 4. Dataset/graph-scale tables — all downstream of Finding 0

`tab:raw_modality_splits`, `tab:summary_stats_from_logs`, and
`tab:graph_hparams_stats` all report the fictitious 759,582-instance,
310-feature, 3M-edge multi-modal scale. Real numbers (both datasets) are in
`results/{toniot,edgeiiotset}/main/dataset_summary.csv` — LaTeX
replacements below. Two additional, real, checkable findings from this
data worth folding in:

- **Window-size (Δt) has no measurable effect on ToN-IoT.** The manuscript
  claims Δt=60s is *"selected on validation"* (implying a meaningful
  sweep). The real window-sensitivity sweep (Δt ∈ {30, 60, 120, 300}s)
  gives **identical** macro-F1 (0.4994) and identical graph topology at
  every candidate value. There was nothing to select — state this as a
  finding, not a tuning decision.
- **Internal reproducibility flag:** the same nominal configuration
  (stacked RF-GCN, Δt=60s, seed=42) yields macro-F1 **0.3089** in the
  3-seed main table but **0.4994** in the single-seed window-sensitivity
  run — driven by LMEP's F1 swinging between 0 and 0.667 because the test
  split contains exactly **one** LMEP-positive asset. This is expected
  instability given ToN-IoT's extreme minority-class sparsity (3/5/11
  total IAD/LMEP/IMP assets), not a pipeline bug, but it means **no
  single-seed ToN-IoT stage-detection number for the rare stages should be
  quoted as representative** — only the 3-seed mean±std, with the caveat
  that the std itself is likely an underestimate at this sample size.

---

## 5. Scalability/implementation paragraph — describes hardware and a training regime that wasn't used

Manuscript §5.7 ("Scalability, training strategy, and system resources")
claims: 32 vCPU + 256GB RAM, **GPU** peak memory 11.8GB, mini-batch
optimization with **neighbor sampling** (fanouts {15,10}, batch 4096),
throughput ~45k nodes/s, 18.2s/epoch GCN training.

The validated pipeline trains the GCN **transductively** (a single
full-graph forward pass — confirmed directly in the final-validation log:
*"Fitting final GCN on full train split (transductive forward pass over
full graph)"*), on CPU (no CUDA/GPU reference anywhere in the codebase or
logs; `README.md` states `torch==2.13 (cpu)`), with no neighbor sampling or
mini-batching code anywhere in `src/`. None of the hardware/throughput
numbers in this paragraph are reproducible from the current implementation.

**Action:** remove this paragraph and replace with the real, measured
`inference_latency.csv` numbers (both datasets) — a smaller but fully
supported and arguably more useful deployment-relevant table (per-component
latency: feature aggregation, RF scoring, GCN forward pass, meta-learner,
end-to-end, throughput).

---

## 6. Training-dynamics figure (`epoch.png`, Figure `fig:epoch`)

No epoch-by-epoch loss/metric log of any kind exists anywhere in
`results/` for either dataset (searched the full tree for `.log`,
`.json`, `.csv` with per-epoch structure — none found; the GCN/GRU
trainers use early stopping internally but do not persist per-epoch
history to disk). The entire paragraph describing "baseline / overfit /
faster-noisier" curves and their qualitative behavior cannot be traced to
any artifact. **Action: flag, do not silently keep.** Either (a) instrument
the trainers to log per-epoch history and regenerate this figure from a
real run, or (b) remove the figure and its paragraph. I cannot verify or
fabricate this one — recommend (a) if it's cheap, since epoch curves would
be genuinely useful evidence of stable optimization, but it is a rerun of
training with new instrumentation, which needs your sign-off since your
instruction was "do not retain unsupported results" not "silently add new
training runs."

---

## 7. Inter-annotator agreement claim (κ = 0.78, consensus = 83.3%)

This is a manual human-annotation exercise (four researchers reviewing the
attack-type→stage mapping), not something the pipeline computes — no file
in `results/` contains an inter-annotator statistic, so I can neither
confirm nor refute it from pipeline artifacts. It's orthogonal to the
model-result claims above. **Action: keep, but flag for the authors to
confirm this annotation exercise actually took place and these are its real
numbers** — it's the one claim in the paper I have no way to check either
direction. Complementary and independently checkable: the real
**stage-mapping sensitivity** experiment (primary/conservative/expanded
mapping variants) is new evidence of mapping robustness from an orthogonal,
fully automated angle — worth adding alongside the κ claim, not replacing
it.

---

## 8. What's fully supported — no change needed

- Threat model, three-stage (IAD/LMEP/IMP) definition, ATT&CK-for-ICS
  mapping methodology (§3) — qualitative/methodological, not data-dependent.
- Asset-disjoint splitting is genuinely implemented: `VALIDATION_REPORT.json`
  `no_asset_in_multiple_splits` — **PASS**, both datasets.
- TRAIN-only feature scaling: `scaler_fit_train_only` — **PASS**, both.
- No test-split leakage into one-vs-rest threshold selection:
  `no_test_leakage_in_threshold_selection` — **PASS**, both (this validates
  the *method*; §3 above shows the resulting ToN-IoT *numbers* are noisy).
- Out-of-fold stacking subgraph is TRAIN-only:
  `oof_subgraph_train_only` — **PASS**, both.
- No IMP-stage evidence reaches the forecaster's input features, and no
  forecasting instance's prefix ever contains an IMP-labeled window:
  `forecaster_pre_cutoff_only`, `no_imp_evidence_in_forecaster_features` —
  **PASS** (both report "N/A — impact forecasting not valid" for
  Edge-IIoTset, and pass cleanly for ToN-IoT's zero-instance case). The
  leakage-*control* claim survives even though the forecasting *result*
  doesn't.
- No synthetic-data marker present; real data confirmed in place —
  `no_synthetic_data` — **PASS**, both.
- The "Scope of generalisation beyond ToN-IoT" / Edge-IIoTset external
  validation framing (§1, red-tracked) — now **better** supported than
  when written: Edge-IIoTset has a complete real main-results table, OvR
  table, and full reviewer-experiment suite (previously it had none).
- Graph-ablation and edge-typing discussion (§4, "Edge-typed variants") —
  the qualitative argument (untyped GCN for scalability, ablation isolates
  interaction vs. temporal) is now backed by real `graph_ablation.csv` for
  both datasets — keep the argument, cite real numbers.
- Baseline-tuning-on-VAL-only claim — now backed by real
  `baseline_tuning.csv` for both datasets.

---

## 9. New tables/figures to add (reviewer-experiment results with no manuscript home yet)

None of these existed in the manuscript in any form before this audit —
they are the direct output of the reviewer-response experiment suite:

1. **Edge-IIoTset main results** — dataset summary, 4-class stage
   detection, OvR alerting. The manuscript promises this dataset's role
   (Table `tab:edgeiiotset_stage_mapping`, prose in §5.1.1) but reports
   **zero** Edge-IIoTset numbers anywhere.
2. **Stage-mapping sensitivity** (both datasets) — 3 mapping variants ×
   macro-F1; complements the κ=0.78 claim with an automated robustness
   check.
3. **Graph ablation** (both datasets) — interaction-only / temporal-only /
   both; directly the ablation §4 already promises but never shows.
4. **Meta-learner ablation** (both datasets) — explains and partially
   mitigates the Finding-1 stacking underperformance.
5. **Baseline-tuning fairness table** (both datasets) — VAL-selected
   hyperparameter grids for RF/XGBoost/LightGBM/GCN/GraphSAGE/GAT.
6. **Bootstrap 95% CI table** (both datasets) — replaces the unsupported
   K=3-split × S=5-seed protocol description with what was actually run
   (asset-level bootstrap, B=1000 requested; ToN-IoT only 50/1000 valid
   iterations due to minority-class sparsity — report this caveat
   explicitly, don't present it as a standard CI).
7. **Inference latency** (both datasets) — replaces §5.7's fabricated
   GPU/mini-batch paragraph.
8. **Window-sensitivity table + figure** (ToN-IoT only; Edge-IIoTset
   validly `SKIPPED_WITH_REASON`) — replaces the "Δt selected on
   validation" claim with the real finding (no measurable effect).
9. **Horizon-sensitivity SKIPPED note** (both datasets) — documents why
   forecasting is off the table, supporting the Finding-2 rewrite.

All source files exist now: `results/{toniot,edgeiiotset}/reviewer_experiments/*.csv`,
`results/manuscript_tables/tab_*.tex` (already LaTeX-formatted, can be
`\input{}`-ed directly or the content pasted inline — see §11), and figures
`results/figures/{toniot,edgeiiotset}/*.png`.

---

## 10. Unsupported claims to remove outright (no salvageable real number)

- Abstract's forecasting sentence (ROC-AUC/PR-AUC/Capture@5%).
- `tab:impact_split_counts`, `tab:impact_results_combined`.
- The impact-prediction row of `tab:seed_robustness`.
- §6.2 paragraphs: "Held-out performance and triage ranking", "Where the
  dual-stream design helps", "Imbalance and training dynamics" (the
  "10/236... 16/243" sentence specifically).
- §5.7's GPU/mini-batch/neighbor-sampling/45k-nodes-per-sec paragraph.
- Figure `fig:epoch` and its paragraph, unless real per-epoch logs are
  generated first (flagged, not auto-removed — your call).
- `tab:raw_modality_splits`, `tab:summary_stats_from_logs` in their current
  multi-modal form.

## 11. Claims to soften (real signal exists, but weaker/different than stated)

- "the proposed stacked RF-GCN achieves the highest Macro-F1... indicating
  complementary signals" → soften to: stacking is proposed and its
  leakage-safe protocol validated; on the two real datasets evaluated it is
  not the top performer as configured (cite real table); the meta-learner
  ablation identifies gradient-boosting as a promising fix, left for future
  tuning rather than claimed as a result.
- "$\Delta t=60$s (selected on validation identities only)" → soften to:
  Δt was swept over {30,60,120,300}s on ToN-IoT and found to have no
  measurable effect on Macro-F1 in this range; 60s retained as the default.
- Inter-annotator κ/consensus sentence → keep number, add "(manual
  annotation exercise, not reproducible from the automated pipeline)" for
  honesty, and reference the new automated stage-mapping-sensitivity table
  as complementary evidence.
- "Errors in Stage~1/Stage~2 detection affect the impact predictor..." →
  this sentence presumes forecasting was evaluated; soften/move to the
  rewritten forecasting-limitations paragraph (§2 above) since forecasting
  itself is now reported as invalid, not merely imperfect.

---

## 12. Ready-to-paste LaTeX

### 12.1 Abstract (replace last two sentences)

Old:
```latex
Under identity-preserving (asset-disjoint) evaluation on ToN-IoT, the stacked detector achieves a macro-F1 of 0.8159 on the four-class setting \{Benign, IAD, LMEP, IMP\}, with minority-stage F1 of 0.7514 (IAD), 0.7305 (LMEP), and 0.8115 (IMP). For impact forecasting, the proposed dual-stream GRU achieves ROC-AUC 0.9047 and PR-AUC 0.4186 and captures 0.6820 of impacted assets within the top 5\% highest-risk predictions, demonstrating practical early-warning triage utility.
```

New:
```latex
Under identity-preserving (asset-disjoint) evaluation on the real ToN-IoT network-flow data (11,536 asset--window instances), the stacked RF--GCN detector achieves a macro-F1 of 0.309, versus 0.368 for the graph-only GCN baseline; on a second, independently evaluated real dataset (Edge-IIoTset, 142,095 instances), RF alone is strongest (macro-F1 0.615) and the stacked model underperforms every baseline (macro-F1 0.165). A meta-learner ablation shows this gap is driven by the stacking meta-learner choice rather than the fusion strategy itself: substituting a gradient-boosting meta-learner recovers macro-F1 to 0.308 on Edge-IIoTset. For impact forecasting, we find the pre-impact forecasting task is not empirically evaluable on either real dataset as currently constituted: ToN-IoT's per-asset activity is too short-lived to construct a valid pre-impact cutoff at any tested horizon, and Edge-IIoTset's published timestamps carry no genuine calendar date. We report this as a data-availability limitation rather than a negative forecasting result, and verify -- independently of whether forecasting can run -- that the leakage-control protocol itself (no post-cutoff or Impact-stage evidence ever reaches the forecaster) holds by construction.
```

### 12.2 Table `tab:main_4class_results_all` (replace body; add Edge-IIoTset panel)

```latex
\begin{table*}[!t]
\centering
\caption{Four-class stage classification on the asset-disjoint test split, real data (mean$\pm$std over 3 seeds; ToN-IoT: network-flow modality, 11,536 instances, 104 features; Edge-IIoTset: 142,095 instances, 175 features, record-level fallback -- see \S\ref{subsec:dataset_description}). Macro-F1 is the primary metric.}
\label{tab:main_4class_results_all}
\renewcommand{\arraystretch}{1.12}
{\tiny
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{Macro-F1} & \textbf{F1(Benign)} & \textbf{F1(IAD)} & \textbf{F1(LMEP)} & \textbf{F1(IMP)} \\
\midrule
\multicolumn{6}{l}{\textit{ToN-IoT}}\\
RF (feature-only) & $0.250\pm0.000$ & $0.999\pm0.000$ & $0.000\pm0.000$ & $0.000\pm0.000$ & $0.000\pm0.000$ \\
GCN (graph-only) & $\mathbf{0.368\pm0.047}$ & $0.998\pm0.001$ & $0.133\pm0.189$ & $0.000\pm0.000$ & $0.340\pm0.047$ \\
Late fusion (avg prob.) & $0.339\pm0.064$ & $0.999\pm0.000$ & $0.167\pm0.236$ & $0.000\pm0.000$ & $0.190\pm0.269$ \\
Stacked RF--GCN (proposed) & $0.309\pm0.047$ & $0.998\pm0.001$ & $0.000\pm0.000$ & $0.000\pm0.000$ & $0.237\pm0.189$ \\
XGBoost & $0.250\pm0.000$ & $0.999\pm0.000$ & $0.000\pm0.000$ & $0.000\pm0.000$ & $0.000\pm0.000$ \\
LightGBM & $0.250\pm0.000$ & $0.999\pm0.000$ & $0.000\pm0.000$ & $0.000\pm0.000$ & $0.000\pm0.000$ \\
GRU (no-graph temporal) & $0.339\pm0.008$ & $0.999\pm0.000$ & $0.000\pm0.000$ & $0.000\pm0.000$ & $0.356\pm0.031$ \\
GraphSAGE & $0.362\pm0.050$ & $0.998\pm0.001$ & $0.133\pm0.189$ & $0.111\pm0.157$ & $0.206\pm0.147$ \\
GAT & $0.353\pm0.074$ & $0.998\pm0.002$ & $0.000\pm0.000$ & $0.000\pm0.000$ & $0.413\pm0.294$ \\
\midrule
\multicolumn{6}{l}{\textit{Edge-IIoTset (external validation)}}\\
\textbf{RF (feature-only)} & $\mathbf{0.615\pm0.003}$ & $1.000\pm0.000$ & $0.424\pm0.007$ & $0.503\pm0.004$ & $0.531\pm0.005$ \\
GCN (graph-only) & $0.348\pm0.136$ & $0.444\pm0.272$ & $0.251\pm0.205$ & $0.315\pm0.069$ & $0.382\pm0.010$ \\
Late fusion (avg prob.) & $0.521\pm0.047$ & $0.852\pm0.110$ & $0.366\pm0.041$ & $0.368\pm0.083$ & $0.499\pm0.019$ \\
Stacked RF--GCN (proposed) & $0.165\pm0.060$ & $0.046\pm0.059$ & $0.170\pm0.159$ & $0.000\pm0.000$ & $0.443\pm0.044$ \\
XGBoost & $0.592\pm0.001$ & $1.000\pm0.000$ & $0.384\pm0.004$ & $0.489\pm0.001$ & $0.497\pm0.001$ \\
LightGBM & $0.582\pm0.003$ & $1.000\pm0.000$ & $0.416\pm0.020$ & $0.461\pm0.004$ & $0.452\pm0.004$ \\
GRU (no-graph temporal) & \multicolumn{4}{l}{N/A -- impact forecasting invalid for this dataset (see \S\ref{sec:impact_discussion})} \\
GraphSAGE & $0.382\pm0.058$ & $0.322\pm0.251$ & $0.467\pm0.014$ & $0.348\pm0.046$ & $0.389\pm0.100$ \\
GAT & $0.336\pm0.141$ & $0.338\pm0.309$ & $0.313\pm0.155$ & $0.327\pm0.088$ & $0.364\pm0.078$ \\
\bottomrule
\end{tabular}
}
\footnotesize{\textit{Note:} On Edge-IIoTset, Stacked RF--GCN underperforms every baseline including plain RF; the meta-learner ablation (Table~\ref{tab:meta_learner_ablation}) attributes this to the logistic-regression meta-learner and shows gradient boosting largely recovers performance (single-seed macro-F1 0.308).}
\end{table*}
```

### 12.3 Table `tab:ovr_combined` (replace; add Edge-IIoTset panel, caveat ToN-IoT)

```latex
\begin{table*}[!t]
\centering
\caption{Stage-wise one-vs-rest alerting on the asset-disjoint test split, real data. ToN-IoT's test split contains only 1, 1, and 3 positive assets for IAD/LMEP/IMP respectively out of 2,307 -- these numbers are not stable operating-point estimates and are reported for completeness only. Edge-IIoTset is well-powered (14.6k--19.5k positives out of 62,990) and is the primary OvR result for this table.}
\label{tab:ovr_combined}
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{lccccc}
\toprule
\textbf{Stage (one-vs-rest)} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{PR-AUC} & \textbf{$n_{\text{pos}}$ / $n_{\text{total}}$} \\
\midrule
\multicolumn{6}{l}{\textit{ToN-IoT (small-count caveat -- not a stable estimate)}}\\
Stage~1 (IAD)  & 0.000 & 0.000 & 0.000 & 0.053 & 1 / 2{,}307 \\
Stage~2 (LMEP) & 0.333 & 1.000 & 0.500 & 0.333 & 1 / 2{,}307 \\
Stage~3 (IMP)  & 0.000 & 0.000 & 0.000 & 0.340 & 3 / 2{,}307 \\
\midrule
\multicolumn{6}{l}{\textit{Edge-IIoTset (well-powered)}}\\
Stage~1 (IAD)  & 0.569 & 0.129 & 0.211 & 0.351 & 19{,}492 / 62{,}990 \\
Stage~2 (LMEP) & 0.078 & 0.081 & 0.080 & 0.157 & 14{,}649 / 62{,}990 \\
Stage~3 (IMP)  & 0.727 & 0.342 & 0.465 & 0.503 & 18{,}723 / 62{,}990 \\
\bottomrule
\end{tabular}
\end{table*}
```

### 12.4 Impact-forecasting section rewrite (replace §6.2 tables + surrounding prose)

```latex
\subsection{Impact Prediction: Data-Availability Limitation}
\label{sec:impact_discussion}

\paragraph{Both real datasets fail the pre-impact instance-construction gate.}
Before reporting forecasting numbers, the pipeline checks whether any valid pre-impact
forecasting instance can be constructed at all (Section~\ref{subsubsec:task_impact_prediction}).
On ToN-IoT, this check fails at every candidate horizon $H\in\{10,30,60\}\times\Delta t$: every
asset's entire observed activity fits inside a single $\Delta t$ window, so no cutoff exists
that leaves both pre-cutoff evidence and a later window to confirm the outcome. On Edge-IIoTset,
the published \texttt{frame.time} field carries a genuine calendar date for 0.0\% of rows (naive
parsing silently defaults every row to January~1st), so no chronological ordering exists to
forecast across. We report both as data-availability limitations rather than as forecasting
failures: the model architecture and leakage-control protocol (Section~\ref{sec:framework_impact})
are implemented and independently verified (no post-cutoff or Impact-stage evidence reaches the
forecaster's input in either case, confirmed by re-deriving the feature-construction pipeline at
runtime), but we do not report forecasting accuracy numbers because none can be honestly computed
on either dataset as currently available. We regard the pipeline's refusal to fabricate a number
here as a methodological strength rather than report placeholder or illustrative figures.

\paragraph{What would make forecasting evaluable.} The gate identifies exactly what is missing:
(i) for ToN-IoT, assets with activity spanning multiple $\Delta t$ windows (the current network-flow
capture is too short-lived per asset); (ii) for Edge-IIoTset, a trustworthy per-record timestamp
with a genuine calendar date. Either a longer-duration capture (ToN-IoT) or a corrected/re-released
timestamp field (Edge-IIoTset) would allow the same, already-implemented forecasting protocol to be
evaluated without any change to the model or leakage controls.
```

Delete `tab:impact_split_counts`, `tab:impact_results_combined`, and the
paragraphs "Split statistics (representative run)", "Imbalance and
training dynamics", "Held-out performance and triage ranking", "Where the
dual-stream design helps (empirical signal)", and "Failure cases and
limitations (when forecasting misses impact)" — all presumed forecasting
ran. Keep "Complementarity with conservative IMP alerting" only if reworded
to not presuppose a forecasting result exists (it currently reads as if
both signals are available side-by-side).

### 12.5 `tab:seed_robustness` (drop the impact-prediction row)

```latex
\begin{table*}[!t]
\centering
\caption{Robustness across seeds (mean$\pm$std) on the asset-disjoint test split, real data.}
\label{tab:seed_robustness}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{lccc}
\toprule
\textbf{Task} & \textbf{Model} & \textbf{ToN-IoT} & \textbf{Edge-IIoTset} \\
\midrule
Stage detection & RF--GCN stacking & Macro-F1: $0.309\pm0.047$ & Macro-F1: $0.165\pm0.060$ \\
Stage detection & RF (best on Edge-IIoTset) & Macro-F1: $0.250\pm0.000$ & Macro-F1: $0.615\pm0.003$ \\
Impact prediction & Dual-stream GRU & \multicolumn{2}{c}{Not evaluable -- see \S\ref{sec:impact_discussion}} \\
\bottomrule
\end{tabular}
\end{table*}
```

### 12.6 New: dataset-summary table (replace `tab:raw_modality_splits` + `tab:summary_stats_from_logs`)

```latex
\begin{table}[!t]
\centering
\caption{Real-data dataset summary, both evaluated datasets. ToN-IoT uses the network-flow modality only (see \S0/\ref{subsec:dataset_description} for scope note); Edge-IIoTset falls back to record-level instances because published timestamps lack a genuine calendar date.}
\label{tab:dataset_summary_real}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{lrr}
\toprule
\textbf{Statistic} & \textbf{ToN-IoT} & \textbf{Edge-IIoTset} \\
\midrule
Raw records & 461{,}043 & 157{,}800 (142{,}095 kept) \\
Unit of analysis & asset--time window ($\Delta t=60$s) & record-level (fallback) \\
Final instances & 11{,}536 & 142{,}095 \\
Distinct assets & 11{,}536 & 17{,}687 \\
Feature dimension & 104 & 175 \\
Train / Val / Test & 6{,}922 / 2{,}307 / 2{,}307 & 68{,}301 / 10{,}804 / 62{,}990 \\
Stage counts (Benign/IAD/LMEP/IMP) & 11{,}517 / 3 / 5 / 11 & 24{,}301 / 41{,}189 / 30{,}775 / 45{,}830 \\
Graph nodes & 11{,}536 & 142{,}095 \\
Graph interaction edges & 11{,}555 & 17{,}685 \\
Graph temporal edges & 0 & 0 \\
Average degree & 2.00 & 0.25 \\
\bottomrule
\end{tabular}
\end{table}
```

### 12.7 New: inference latency table (replaces §5.7 GPU paragraph)

```latex
\begin{table}[!t]
\centering
\caption{Real, measured inference latency (CPU; per-instance mean over 30 repeats), both datasets.}
\label{tab:inference_latency}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{lrr}
\toprule
\textbf{Component} & \textbf{ToN-IoT (ms)} & \textbf{Edge-IIoTset (ms)} \\
\midrule
Feature aggregation & 17.6 & 12.0 \\
RF scoring & 153.2 & 126.8 \\
GCN full-graph forward pass & 12.7 & 98.4 \\
GCN, amortized per test node & 0.0055 & 0.0016 \\
Stacking meta-learner & 0.80 & 0.20 \\
\midrule
End-to-end (per instance) & \textbf{171.6} & \textbf{139.0} \\
Throughput (instances/sec) & 5.83 & 7.20 \\
\bottomrule
\end{tabular}
\end{table}
```

### 12.8 New: meta-learner ablation, graph ablation, baseline tuning, bootstrap CI, stage-mapping sensitivity, window sensitivity

These already exist as publication-formatted LaTeX in
`results/manuscript_tables/tab_{toniot,edgeiiotset}_{meta_learner_ablation,
graph_ablation,baseline_tuning,bootstrap_ci,stage_mapping_sensitivity,
window_sensitivity,horizon_sensitivity}.tex` — generated directly from the
same validated CSVs cited throughout this document. Recommended: `\input{}`
them directly rather than retyping, e.g.:

```latex
\input{manuscript_tables/tab_toniot_graph_ablation.tex}
\input{manuscript_tables/tab_edgeiiotset_graph_ablation.tex}
\input{manuscript_tables/tab_toniot_meta_learner_ablation.tex}
\input{manuscript_tables/tab_edgeiiotset_meta_learner_ablation.tex}
\input{manuscript_tables/tab_toniot_baseline_tuning.tex}
\input{manuscript_tables/tab_edgeiiotset_baseline_tuning.tex}
\input{manuscript_tables/tab_toniot_bootstrap_ci.tex}
\input{manuscript_tables/tab_edgeiiotset_bootstrap_ci.tex}
\input{manuscript_tables/tab_toniot_stage_mapping_sensitivity.tex}
\input{manuscript_tables/tab_edgeiiotset_stage_mapping_sensitivity.tex}
\input{manuscript_tables/tab_toniot_window_sensitivity.tex}
\input{manuscript_tables/tab_edgeiiotset_window_sensitivity.tex} % SKIPPED note
\input{manuscript_tables/tab_toniot_horizon_sensitivity.tex}     % SKIPPED note
\input{manuscript_tables/tab_edgeiiotset_horizon_sensitivity.tex} % SKIPPED note
```

Corresponding figures for §12.9 (window/graph/meta-learner sensitivity
plots) are at `results/figures/{toniot,edgeiiotset}/{window_sensitivity,
graph_ablation,meta_learner_ablation}.png` (no Edge-IIoTset
window-sensitivity plot — validly skipped).
