# Stage-Aware AI Security Assurance for IIoT

[![CI](https://github.com/mahend72/iiot-ai-security-assurance/actions/workflows/ci.yml/badge.svg)](https://github.com/mahend72/iiot-ai-security-assurance/actions/workflows/ci.yml)

This repository is an AI security assurance study: it evaluates how
trustworthy graph- and feature-based machine learning models are for staged
intrusion detection in Industrial IoT (IIoT) environments, under an
evaluation protocol designed to prevent the model from being right for the
wrong reasons.

The intrusion-detection task itself — classifying network activity into an
ATT&CK-inspired attack-stage taxonomy — is the vehicle. The object under
test is the evaluation: whether a reported score reflects genuine
detection skill under identity-preserving, leakage-controlled conditions,
or an artifact of how the data was split, scaled, or windowed. The
repository documents one case (a timestamp-resolution bug in the ToN-IoT
windowing code) where the difference was the entire result — see
[Reproducibility and Data Correction](#reproducibility-and-data-correction).

Two published IIoT/IoT security benchmarks are used: **ToN-IoT** (network
flows) and **Edge-IIoTset**. Both datasets are run through identical
pipeline code, driven entirely by per-dataset YAML configuration — no
dataset-specific logic is hardcoded.

## Why This Is an AI Security Assurance Project

Model comparison papers routinely report a "winning" architecture without
verifying that the evaluation itself is trustworthy. This repository treats
that verification as the primary deliverable, not an afterthought:

- **Leakage-controlled evaluation.** Every split, scaling operation, graph
  construction, and hyperparameter selection is checked structurally
  against the possibility of train/test contamination — not just assumed
  correct because the code "looks right." See
  [Assurance and Leakage Controls](#assurance-and-leakage-controls).
- **Evidence-based model comparison instead of a fixed conclusion.** The
  results are reported as measured, including cases where the model the
  pipeline was originally built to showcase (a stacked RF–GCN detector)
  underperforms simpler baselines. See [Main Results](#main-results).
- **A methodology retained even when the data can't support it.** The
  pre-impact forecasting component is fully implemented and leakage-safe,
  but the corrected data does not provide a statistically valid held-out
  evaluation. That is reported as a data-availability limitation, not
  hidden or worked around by relaxing the split protocol. See
  [Pre-Impact Forecasting Methodology](#pre-impact-forecasting-methodology).
- **A documented instance of the evaluation catching its own error.** A
  windowing bug silently collapsed ToN-IoT into a near-degenerate dataset;
  every affected number was regenerated, and the before/after diff is kept
  in the repository rather than quietly overwritten. See
  [Reproducibility and Data Correction](#reproducibility-and-data-correction).

## Security Problem and Threat Model

IIoT environments (industrial control networks, edge gateways, SCADA-adjacent
assets) are attacked in stages: reconnaissance and initial access, lateral
movement and privilege escalation, then an impact action (denial of service,
destructive payload, ransomware). A detector that only classifies
benign-vs-attack cannot distinguish "an asset is being probed" from "an
asset is about to be taken down" — both collapse to "attack." A stage-aware
detector preserves that distinction, which is what an operator needs to
prioritize response.

The datasets used here are IoT/IIoT cybersecurity benchmarks — labeled
captures of network traffic and attack activity — not live telemetry from a
real multi-organizational industrial supply chain. Results should be read
as evidence about model behavior on these benchmarks, not as a validated
claim about a production IIoT deployment or a full supply-chain compromise
scenario.

## ATT&CK-Inspired Stage Model

Raw per-dataset attack labels are mapped to four operational stages,
loosely aligned with the early-to-late progression in MITRE ATT&CK
(reconnaissance/initial-access → lateral-movement/persistence → impact):

| Stage | Meaning | Example raw labels (ToN-IoT) |
|---|---|---|
| **Benign** | Normal traffic | `normal` |
| **IAD** — Initial Access & Discovery | Reconnaissance, credential/entry attempts | `scanning`, `password`, `xss` |
| **LMEP** — Lateral Movement / Escalation / Persistence | Post-compromise pivoting and persistence | `backdoor`, `injection`, `mitm` |
| **IMP** — Impact | Availability or destructive impact | `dos`, `ddos`, `ransomware` |

The mapping is not hardcoded in Python: it lives in
`configs/stage_mapping_toniot.yaml` and
`configs/stage_mapping_edgeiiotset.yaml`, with the rationale for each
raw-label → stage assignment written inline so it is auditable and
editable. Unmapped raw values are logged as warnings rather than silently
dropped or defaulted. Two alternative mapping variants (`_conservative`,
`_expanded`) exist per dataset and are used for the stage-mapping
sensitivity check below — the four-stage taxonomy is a modeling choice, not
a fixed ground truth, and the sensitivity result should be read alongside
any headline number.

## Assurance and Leakage Controls

These controls are implemented in code, not just described. Each row names
where to look and, where available, the automated check that verifies it.

| Control | Implementation | Verified by |
|---|---|---|
| Asset-disjoint splitting | `src/preprocessing/splitting.py` splits by unique asset identity, never by record | `no_asset_in_multiple_splits` |
| Asset-disjoint cross-validation | Out-of-fold stacking features use `sklearn.GroupKFold` grouped by asset (`src/training/stage_detector_trainer.py`) | same asset-disjoint check applied per fold |
| Preprocessing fit on TRAIN only | Feature scaling/encoding (`src/preprocessing/features.py`) fits only on the TRAIN split | `scaler_fit_train_only` |
| No cross-split graph leakage | The GCN used to produce out-of-fold stacking features is trained on a subgraph induced over TRAIN nodes only (`AssetTimeGraph.induced_subgraph`) — val/test node *features*, not just labels, are unreachable during that phase | `oof_subgraph_train_only` |
| Pre-impact temporal cutoffs | `src/training/sequence_builder.py` guarantees no window at or after an asset's first IMP-labeled window appears in a forecasting input sequence | `forecaster_pre_cutoff_only` |
| IMP evidence excluded from forecasting inputs | `src/training/impact_forecast_trainer.py` drops the IMP probability dimension from the stage-evidence features it consumes | `no_imp_evidence_in_forecaster_features` |
| No test-time threshold selection | One-vs-rest alerting thresholds are selected on VAL, applied to TEST | `no_test_leakage_in_threshold_selection` |
| Baseline tuning on validation only | `run_reviewer_experiments.py`'s hyperparameter grid for RF/XGBoost/LightGBM/GCN/GraphSAGE/GAT is selected on VAL, scored on TEST | `results/*/reviewer_experiments/baseline_tuning.csv` |
| Dataset/schema provenance | Adapters (`src/data/generic_adapter.py`) inspect the real CSV header and match against config-declared candidates; a missing required field raises, never guesses | — |
| Synthetic-data guard | Any script refuses to run on data still carrying the synthetic-generator marker (`src/utils/data_provenance.py`) | `no_synthetic_data` |
| Provenance stamps | Every results file is stamped with dataset/config/git/timestamp metadata | `provenance_stamps_present` |

Robustness and sensitivity analyses (not leakage controls per se, but part
of the same assurance posture):

- **Stage-mapping sensitivity** — primary/conservative/expanded label
  mappings, to check the headline result isn't an artifact of one taxonomy
  choice.
- **Graph-construction ablation** — interaction-edges-only,
  temporal-edges-only, both.
- **Window-size sensitivity** — Δt swept across candidate values.
- **Meta-learner ablation** — logistic regression / MLP / gradient boosting
  as the stacking combiner.
- **Repeated seeds** — main results are mean ± std over 3 seeds.
- **Bootstrap confidence intervals** — asset-level resampling (not
  record-level, to respect the same identity boundary as the splits).
- **Inference-latency measurement** — per-component and end-to-end timing.

All 16 checks (8 categories × 2 datasets) currently pass:
`results/VALIDATION_REPORT.json`.

## Continuous Integration

The [`CI` workflow](.github/workflows/ci.yml) runs on every push and pull
request to `main`, on Python 3.10 and 3.11. It checks **code integrity and
assurance safeguards**, not production readiness — there is no deployment
target for this project. Each run:

1. Byte-compiles every module in `src/` and `scripts/` to catch syntax
   errors early.
2. Runs the `pytest` suite in `tests/`, including regression tests for the
   ToN-IoT windowing fix (see
   [Reproducibility and Data Correction](#reproducibility-and-data-correction)),
   a dedicated assurance-property suite
   (`tests/test_assurance_properties.py`) that re-runs several of the
   [Assurance and Leakage Controls](#assurance-and-leakage-controls) checks
   above — asset-disjoint splitting, no identity leakage across partitions,
   train-only preprocessing fitting, no IMP-stage evidence reaching the
   forecaster, no cross-split graph/neighbourhood leakage, valid
   stage-mapping configuration, and the synthetic-data guard — against a
   small in-memory synthetic dataset, and a generator-safety regression
   suite (`tests/test_generate_synthetic_data_safety.py`, see below), all
   without any real data present.
3. Generates a small synthetic dataset into the dedicated `data/synthetic/`
   directory (`python scripts/generate_synthetic_data.py --out-root
   data/synthetic ...` — never `data/raw/`) and runs the stage-detection
   pipeline end-to-end on it (`python scripts/run_stage_detection.py
   --dataset toniot --raw-dir data/synthetic/toniot ...`) as a smoke test
   that the full pipeline still executes.

**Synthetic-data overwrite safety.** `scripts/generate_synthetic_data.py`
refuses to overwrite any target file that is not itself prior output of
the generator (i.e. that has no `.SYNTHETIC_DATA_MARKER` next to it) —
this is what stops it from ever silently clobbering a real dataset placed
in `data/raw/<dataset>/`. Overwriting such a file requires an explicit
`--force` flag and prints a loud warning when used. `--out-root` selects
where synthetic data is written (default `data/raw`, matching the local
smoke-test workflow below; CI uses a dedicated `data/synthetic` instead).
See `tests/test_generate_synthetic_data_safety.py` for the regression
tests pinning this behavior.

**CI does not use, download, or require the real ToN-IoT or Edge-IIoTset
datasets**, and its synthetic-data smoke test carries no scientific meaning
— it does not reproduce, and is not a substitute for, the
[Main Results](#main-results) reported below. Those come only from the real
datasets run locally, gated by the synthetic-data guard described above.

Tagged releases (`vX.Y.Z`) additionally trigger the
[`Release` workflow](.github/workflows/release.yml), which re-runs the same
lightweight validation against the tagged commit and publishes a
reproducible source archive as a GitHub Release.

## Architecture

The pipeline (`src/pipeline.py::prepare_dataset`) is shared, unmodified,
across both datasets:

1. **Load** — schema-inspecting adapter reads the real CSV header and
   matches it against config-declared candidate columns
   (`src/data/generic_adapter.py`, `src/data/toniot_adapter.py`,
   `src/data/edgeiiotset_adapter.py`).
2. **Map** — raw attack labels → {Benign, IAD, LMEP, IMP}
   (`src/mapping/label_mapper.py`).
3. **Split** — asset-disjoint train/val/test partition, decided before any
   downstream step (`src/preprocessing/splitting.py`).
4. **Window** — asset-time-window instances at configurable Δt
   (`src/preprocessing/windowing.py`). Where per-record timestamps are not
   trustworthy for a dataset (see Edge-IIoTset caveat below), the pipeline
   falls back to one instance per raw record rather than fabricating a time
   axis.
5. **Feature** — tabular feature aggregation, scaled/encoded on TRAIN only
   (`src/preprocessing/features.py`).
6. **Graph** — asset-time interaction graph: *interaction edges* connect
   assets active in the same window; *temporal edges* connect an asset's
   consecutive observed windows (`src/graph/graph_builder.py`).
7. **Model** — stage detector(s) trained on the graph/features (see below).
8. **Evaluate** — per-class and macro metrics, bootstrap CIs, plots, CSV/JSON
   export (`src/evaluation/`).

## Models Evaluated

These are compared as alternative security-learning models on the same
leakage-controlled splits — not presented as a single proposed architecture
with the others as strawmen. The strongest model differs by dataset (see
[Main Results](#main-results)).

| Model | Type | Source |
|---|---|---|
| Random Forest | Tabular, feature-only | `src/models/random_forest_model.py` |
| XGBoost | Tabular, gradient boosting | `src/models/gradient_boosting_models.py` |
| LightGBM | Tabular, gradient boosting | `src/models/gradient_boosting_models.py` |
| GCN | Graph-only | `src/models/gcn_model.py` |
| GraphSAGE | Graph-only | `src/models/gcn_model.py` (`conv_type="sage"`) |
| GAT | Graph-only | `src/models/gcn_model.py` (`conv_type="gat"`) |
| GRU (no-graph temporal) | Sequential, feature-only | `src/models/temporal_stage_model.py` |
| Late fusion (RF + GCN) | Score-level combination | `src/training/model_comparison.py` |
| Stacked RF–GCN | Out-of-fold stacking, configurable meta-learner (logistic regression / MLP / gradient boosting) | `src/training/stage_detector_trainer.py`, `src/models/meta_learner.py` |

## Datasets

Neither dataset is bundled (license + size). Download and place:

| Dataset | Expected file | Place in |
|---|---|---|
| ToN-IoT (Network) | `Train_Test_Network.csv` | `data/raw/toniot/` |
| Edge-IIoTset | `ML-EdgeIIoT-dataset.csv` | `data/raw/edgeiiotset/` |

- ToN-IoT: https://research.unsw.edu.au/projects/toniot-datasets
- Edge-IIoTset: Ferrag et al., IEEE Access 2022 (hosted via IEEE DataPort /
  Kaggle mirrors — search "Edge-IIoTset")

If the exact filename isn't found, the adapter falls back to globbing
`data/raw/<dataset>/*.csv`, so multiple CSV shards also work.

| | ToN-IoT | Edge-IIoTset |
|---|---|---|
| Raw records | 461,043 | 157,800 (142,095 kept after cleaning) |
| Unit of analysis | asset-time-window (Δt = 60s) | record-level (see caveat below) |
| Final instances | 34,435 | 142,095 |
| Distinct assets | 11,536 | 17,687 |
| Feature dimension | 126 | 175 |
| Stage counts (Benign/IAD/LMEP/IMP) | 33,808 / 86 / 298 / 243 | 24,301 / 41,189 / 30,775 / 45,830 |
| Graph | 34,435 nodes, 10,168 interaction edges, 22,899 temporal edges | 142,095 nodes, 17,685 interaction edges, 0 temporal edges |

**Edge-IIoTset timestamp caveat:** the published `frame.time` column
resolves to a genuine calendar date for 0.0% of rows in this dataset
(naive parsing defaults everything to Jan-1 — confirmed by the adapter's
own data-quality log, not a pipeline bug). Consequently: asset-level
time-windowing is rejected (it would collapse nearly all assets into a
single degenerate window), temporal graph edges are not constructed, and
window-size/horizon sensitivity and impact forecasting are all gated off
for Edge-IIoTset with a written reason rather than silently producing
numbers from an untrustworthy time axis.

**Synthetic data.** No real data yet, or just want to see the pipeline run?

```bash
python scripts/generate_synthetic_data.py --n-assets 40 --hours 6 --rate-hz 0.05
```

⚠️ **Synthetic data is for smoke-testing the code only.** Results produced
on it have no scientific meaning — every script will still happily produce
metrics/figures on it, but they describe the synthetic generator, not
either real dataset. Every results-generating script refuses to run
(`SyntheticDataGuardError`) if the synthetic-data marker is still present
in `data/raw/<dataset>/` — delete it and place the real CSV before drawing
any conclusion.

The generator itself also refuses to **overwrite** a file in
`data/raw/<dataset>/` that isn't already its own prior output (i.e. that
has no `.SYNTHETIC_DATA_MARKER` next to it) — so running the command above
can never silently clobber a real dataset you've already placed there.
Pass `--force` to override this (only if you're certain the target isn't
real data), or `--out-root <dir>` to write elsewhere entirely.

## Main Results

Stage-detection macro-F1, mean over 3 seeds, on the asset-disjoint test
split. Source: `results/{toniot,edgeiiotset}/main/stage_detection_main.csv`.

| Model | ToN-IoT | Edge-IIoTset |
|---|---|---|
| RF (feature-only) | 0.455 | **0.615** |
| XGBoost | 0.378 | 0.592 |
| LightGBM | 0.535 | 0.582 |
| GCN (graph-only) | 0.660 | 0.348 |
| GraphSAGE | 0.671 | 0.382 |
| **GAT** | **0.687 ± 0.041** | 0.336 |
| GRU (no-graph temporal) | 0.430 | N/A |
| Late fusion (RF + GCN) | 0.658 | 0.521 |
| Stacked RF–GCN (logistic-regression meta-learner) | 0.395 ± 0.067 | 0.165 |

**The strongest architecture is dataset-dependent, not a fixed
conclusion:**

- On corrected ToN-IoT, **GAT is the strongest stage detector**
  (Macro-F1 ≈ 0.687 ± 0.041), with GraphSAGE and GCN also performing
  strongly. Graph-based models substantially outperform tabular baselines
  here.
- On Edge-IIoTset, **Random Forest is strongest** (Macro-F1 ≈ 0.615).
  Graph-based models underperform tabular baselines on this dataset — a
  plausible driver is Edge-IIoTset's unusable timestamp field, which
  forces a record-level graph with no temporal edges and very low average
  degree (0.25), giving the GNNs far less structure to exploit than on
  ToN-IoT.
- **Graph learning is not universally superior.** The two datasets rank
  the same model families in opposite order.
- The default stacked RF–GCN detector (logistic-regression meta-learner)
  **underperforms the strongest graph model on both datasets**. A
  meta-learner ablation (`results/*/reviewer_experiments/meta_learner_ablation.csv`)
  shows switching the combiner to gradient boosting improves stacking
  substantially — ToN-IoT 0.395 → 0.468, Edge-IIoTset 0.165 → 0.308 — but
  even then, gradient-boosting-stacked RF–GCN does **not** outperform the
  strongest standalone model on either dataset (GAT at 0.687 on ToN-IoT,
  RF at 0.615 on Edge-IIoTset). The underperformance is a meta-learner
  choice effect, not evidence that stacking as an architecture is
  unworkable — but it is also not evidence that stacking is the superior
  choice here.

One-vs-rest per-stage alerting (threshold selected on VAL, applied to
TEST): `results/{toniot,edgeiiotset}/main/one_vs_rest_alerting.csv`. On
ToN-IoT, IAD alerting remains weak (F1 0.009, n=48 positives) even at
real class support — a genuine detection-difficulty finding, since IAD
traffic overlaps heavily with benign background noise, not a sample-size
artifact. LMEP (F1 0.570, n=116) and IMP (F1 0.519, n=33) are usable.

Additional sensitivity findings (`results/*/reviewer_experiments/`):

- **Window-size sensitivity (ToN-IoT):** macro-F1 is non-monotonic in Δt —
  30s → 0.376, 60s (default) → 0.308, 120s → 0.579, 300s → 0.532. 60s is
  retained as the default configuration for consistency with the
  forecasting-horizon parameterization, not because it is the
  best-performing value; 120s is flagged as a stronger operating point.
- **Graph ablation (ToN-IoT):** interaction-edges-only (0.359) slightly
  outperforms the combined graph (0.308); temporal-edges-only (0.283) is
  a real, non-trivial signal on its own. The current unweighted union of
  edge types is not strictly additive.
- **Bootstrap 95% CI (ToN-IoT, logistic-regression stack):**
  [0.230, 0.408] around a point estimate of 0.308, 1000/1000 valid
  asset-level resamples.
- **Inference latency:** ToN-IoT 185.5ms/instance end-to-end (RF scoring
  dominates at 167.8ms); Edge-IIoTset 139.0ms/instance.

All 16 reviewer-response experiments (8 categories × 2 datasets) are
either complete with real numbers or validly gated off with a documented,
data-driven reason — see `results/MANUSCRIPT_RESULT_SUMMARY.md` for the
full table.

## Pre-Impact Forecasting Methodology

The repository implements a second, methodologically distinct component: a
dual-stream GRU (`src/models/gru_forecaster.py`) that predicts whether an
asset reaches IMP within a horizon H, conditioned on IAD/LMEP evidence
only. Its leakage controls are real and independently checked (see
[Assurance and Leakage Controls](#assurance-and-leakage-controls)):
`sequence_builder.py` guarantees no window at or after an asset's first
IMP-labeled window can enter a training sequence, and the IMP probability
dimension is dropped from the features the forecaster consumes even where
present.

**This methodology is not currently backed by a statistically valid
held-out evaluation on either dataset:**

- **ToN-IoT:** the corrected windowing produces 21,313 candidate
  forecasting instances, of which only 5 are positive — and all 5 fall in
  TRAIN. VAL (0/1,391) and TEST (0/798) contain zero positive instances.
  ROC-AUC, PR-AUC, and Capture@k% are undefined without both classes
  present in the evaluation split, so none are reported. The root cause is
  a genuine property of the dataset, not a bug: of the 11 assets that ever
  reach IMP, 7 have IMP as their literal first observed window (instant-onset
  attack bursts with no earlier escalation history to forecast from); of
  the 4 assets with real pre-impact history, the asset-disjoint split
  happens to place all of them in TRAIN.
- **Edge-IIoTset:** the timestamp field is not usable (see the dataset
  caveat above), so pre-impact sequences cannot be constructed at all; the
  forecasting pipeline reports `SKIPPED_WITH_REASON` rather than producing
  numbers from an invalid time axis.

Accordingly, this README and the accompanying manuscript **do not report
forecasting ROC-AUC, PR-AUC, or F1 as a validated scientific result** on
either dataset. The forecasting pipeline is retained as a methodological
and research extension — implemented, leakage-controlled, and tested
(`tests/test_windowing.py`, `tests/test_toniot_windowing_integration.py`) —
for longitudinal IIoT datasets with sufficient held-out positive
escalation episodes to support the same asset-disjoint evaluation
protocol used elsewhere in this repository. Numbers seen in
`results/*/impact_forecasting/` under directories timestamped before the
real data was placed are synthetic-generator artifacts (see next section)
and must not be cited.

## Reproducibility and Data Correction

**ToN-IoT temporal-windowing bug (fixed 2026-08-20).**
`src/preprocessing/windowing.py::assign_window_id` divided an
already-in-seconds `datetime64[s]` integer timestamp by `1e9` a second
time — a resolution assumption that broke under this environment's pandas
version (3.0.1 returns `datetime64[s]`, not `datetime64[ns]`, for this
conversion path). The effect: `window_id` collapsed to a single constant
value for every ToN-IoT record, silently turning "one instance per asset
per real 60-second window" into "one instance per asset, aggregated over
its entire ~27-day observed history." This shrank 461,043 raw records to
11,536 degenerate instances (== asset count) with only 3 IAD / 5 LMEP / 11
IMP non-Benign instances surviving in total, made temporal graph edges
structurally impossible (every asset had exactly one window), and produced
a "window size has no effect" finding that was itself a symptom of the bug.

**All ToN-IoT results in this README, in `results/toniot/`, and in the
manuscript were regenerated after the fix**, using the corrected code
against the same unmodified raw file
(`data/raw/toniot/Train_Test_Network.csv`). Pre-fix numbers are archived
under `results/archive/toniot_pre_window_fix_INVALID/` for audit purposes
only and must not be cited or restored. Edge-IIoTset was never affected —
its record-level fallback path never calls `assign_window_id`.

Regression tests pin the fix: `tests/test_windowing.py`,
`tests/test_toniot_windowing_integration.py`. Full before/after diff for
every affected number: `results/TONIOT_WINDOW_FIX_RESULT_DIFF.md`.
Root-cause trace: `results/archive/toniot_pre_window_fix_INVALID/README.md`.
Structural validation after the fix: `results/VALIDATION_REPORT.json`
(16/16 checks passed, both datasets, re-run 2026-08-20).

A second, unrelated caveat: `results/toniot/stage_detection/`,
`results/toniot/impact_forecasting/`, `results/toniot/sensitivity/`, and
the equivalent Edge-IIoTset directories under those same names were written
before the real CSVs were placed on disk and reflect the synthetic
smoke-test generator, not real data — this is the exact scenario the
synthetic-data warning above describes. The current, real-data results
live under `results/{toniot,edgeiiotset}/main/` and
`results/{toniot,edgeiiotset}/reviewer_experiments/`.

## Installation

```bash
cd iiot-ai-security-assurance
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Requires Python 3.10+. Tested with `torch==2.13 (cpu)` and
`torch_geometric==2.8`. The gradient-boosting baselines additionally
require `xgboost` and `lightgbm` (not pinned in `requirements.txt`;
install separately if you intend to run those baselines).

## Data Setup

See [Datasets](#datasets) above for download locations, expected filenames,
and the synthetic-data smoke-test path.

## Running Experiments

Each stage can be run independently, or all at once:

```bash
# Everything, both datasets (stage detection + forecasting + sensitivity sweeps)
python scripts/run_all.py

# Faster run (fewer CV folds) -- good for a first smoke test
python scripts/run_all.py --quick

# One dataset only, skip the slow reviewer-response sweeps
python scripts/run_all.py --datasets toniot --skip-sensitivity
```

Or invoke each stage directly:

```bash
# Stage detection (RF + GCN + stacking meta-learner)
python scripts/run_stage_detection.py --dataset toniot
python scripts/run_stage_detection.py --dataset edgeiiotset

# Impact forecasting (dual-stream GRU). Checks a data-quality/instance-level
# validity gate first and WRITES A REPORT explaining why it was skipped if
# the gate fails, instead of producing an unreportable number -- see
# Pre-Impact Forecasting Methodology above.
python scripts/run_impact_forecasting.py --dataset toniot
python scripts/run_impact_forecasting.py --dataset edgeiiotset

# Reviewer-response / sensitivity experiments (window/horizon sensitivity,
# graph ablation, meta-learner ablation, baseline tuning, inference latency)
python scripts/run_sensitivity.py --dataset toniot
python scripts/run_sensitivity.py --dataset edgeiiotset
```

The tables and figures currently reported in this README and in
`results/manuscript_tables/` were generated with the manuscript-oriented
entry points, which write to `results/<dataset>/main/` and
`results/<dataset>/reviewer_experiments/` instead:

```bash
python scripts/run_main_results.py --dataset toniot --seeds 42 43 44
python scripts/run_reviewer_experiments.py --dataset toniot --seed 42
python scripts/run_final_validation.py --datasets toniot edgeiiotset
```

Useful flags (most scripts): `--graph-mode {interaction_only,temporal_only,both}`,
`--meta-learner {logistic_regression,mlp,gradient_boosting}`, `--n-folds`,
`--delta-t` (window size override), `--seed`. Run `--help` on any script for
the full list.

## Configuration

Everything data- or model-specific lives in YAML, not in code:

- `configs/toniot.yaml`, `configs/edgeiiotset.yaml` — dataset schema
  candidates (columns the adapter looks for), window size, forecasting
  horizon, split fractions, graph construction, and every model
  hyperparameter.
- `configs/stage_mapping_toniot.yaml`, `configs/stage_mapping_edgeiiotset.yaml`
  (plus `_conservative` / `_expanded` variants) — raw attack-type →
  {Benign, IAD, LMEP, IMP} mapping, with the rationale for each mapping
  documented inline.

To change a hyperparameter, edit the relevant YAML — no code changes
needed. `src/utils/config.override()` produces a modified in-memory copy
without touching the file (used internally, e.g., by the window-size sweep
in `run_reviewer_experiments.py`).

## Results and Outputs

```
results/
  toniot/
    main/                          # current, real-data results (post windowing-fix)
      dataset_summary.csv
      stage_detection_main.csv
      one_vs_rest_alerting.csv
      impact_forecasting_SKIPPED_WITH_REASON.json
    reviewer_experiments/          # sensitivity/ablation/tuning/latency, real data
      window_sensitivity.csv, graph_ablation.csv, meta_learner_ablation.csv,
      baseline_tuning.csv, bootstrap_confidence_intervals.csv,
      inference_latency.csv, stage_mapping_sensitivity.csv, ...
    stage_detection/, impact_forecasting/, sensitivity/
      # PRE-REAL-DATA / SYNTHETIC-ERA — do not cite, see
      # Reproducibility and Data Correction above
  edgeiiotset/
    ... same layout ...
  manuscript_tables/                # LaTeX tables generated from the above
  figures/                          # confusion matrices, macro-F1 ranking,
                                     # PR curves, sensitivity plots
  VALIDATION_REPORT.json            # 16/16 structural assurance checks
  MANUSCRIPT_RESULT_SUMMARY.md      # narrative source-of-truth for every number above
  TONIOT_WINDOW_FIX_RESULT_DIFF.md  # full before/after diff for the windowing fix
```

## Project Structure

```
src/
  data/            dataset schema, base + per-dataset adapters, loader factory
  preprocessing/   asset-disjoint split, entity-time-window builder, feature scaling
  mapping/         raw attack-type -> {Benign, IAD, LMEP, IMP} mapping
  graph/           asset-time interaction graph construction (+ ablation modes)
  models/          RandomForest/XGBoost/LightGBM wrappers, GCN/GraphSAGE/GAT,
                    dual-stream GRU, meta-learner
  training/        GCN trainer, stage-detector (RF+GCN+stacking) trainer,
                    forecasting sequence builder + trainer
  evaluation/      metrics (stage + forecasting), bootstrap CIs, plotting, palette
  utils/           config loading, logging, seeding, data provenance guard
  pipeline.py      shared load->map->split->window->graph orchestration
configs/
  toniot.yaml, edgeiiotset.yaml                       dataset schema + hyperparameters
  stage_mapping_toniot.yaml, stage_mapping_edgeiiotset.yaml (+ variants)
scripts/
  generate_synthetic_data.py    smoke-test data generator (NOT real data)
  run_stage_detection.py, run_impact_forecasting.py, run_sensitivity.py, run_all.py
  run_main_results.py, run_reviewer_experiments.py, run_final_validation.py
  generate_reviewer_figures.py, generate_windowing_fix_audit.py
results/           all metrics/CSVs/figures land here, per dataset
tests/             regression tests, including the windowing-fix pin
manuscript/        current_manuscript.tex
```

## Limitations

- **Forecasting is unvalidated on both current datasets.** See
  [Pre-Impact Forecasting Methodology](#pre-impact-forecasting-methodology).
  This is a data-availability limitation, not a claim about the GRU
  architecture's effectiveness.
- **IAD detection is weak on ToN-IoT** (F1 ≈ 0.009 at real class support)
  — reconnaissance-stage traffic overlaps heavily with benign background
  noise in this dataset; this is reported as a real detection-difficulty
  finding, not filtered out.
- **Minority-class support is limited on ToN-IoT** even post-fix (86 IAD /
  298 LMEP / 243 IMP instances out of 34,435) — per-class metrics for
  these stages should be read as indicative, and bootstrap CIs are
  reported specifically to communicate that uncertainty rather than a
  single point estimate.
- **Edge-IIoTset's timestamp field is unusable as published**, which
  removes temporal graph structure and forecasting entirely for that
  dataset — a property of the public CSV, confirmed by the adapter's
  data-quality gate, not a pipeline defect.
- **The four-stage taxonomy is one modeling choice among several** —
  stage-mapping sensitivity shows a meaningful macro-F1 spread across
  primary/conservative/expanded label-mapping variants on ToN-IoT (0.308 /
  0.338 / 0.414).
- **Graph edge types are combined by unweighted union**, not learned
  weighting — the ablation shows this union is not strictly additive on
  ToN-IoT. Directed/typed edge variants are not implemented.
- **The datasets are benchmark captures, not live multi-organizational
  supply-chain telemetry** — see
  [Security Problem and Threat Model](#security-problem-and-threat-model).

## Citation / Related Paper

A manuscript describing this work is maintained at
`manuscript/current_manuscript.tex`. As of this README revision, the
manuscript's ToN-IoT results section is under active reconciliation with
the corrected, single-modality (network-flow-only) pipeline this
repository actually implements and reports above; see
`results/POST_FIX_MANUSCRIPT_CHANGE_LIST.md` for the audit trail. Cite the
manuscript once that reconciliation is complete, or cite this repository
directly in the interim.
