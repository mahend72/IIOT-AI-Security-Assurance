# Multi-stage Attack Detection and Prediction in IIoT Supply Chains Using Graph Neural Networks

A reproducible implementation of a stage-aware intrusion detection framework for Industrial IoT (IIoT) supply chains, run on **ToN-IoT** (main dataset) and
**Edge-IIoTset** (second dataset), on identical pipeline code driven entirely by per-dataset YAML config.


The pipeline:

1. Loads and preprocesses heterogeneous telemetry via a **schema-inspecting dataset adapter** (never assumes a column exists — it reads the real CSV
   header and matches it against config-provided candidates).
2. Maps raw attack labels into four stages: **Benign, IAD** (Initial Access & Discovery), **LMEP** (Lateral Movement / Escalation / Persistence), **IMP**
   (Impact).
3. Builds **asset-time-window instances** at a configurable window size Δt.
4. Constructs an **asset-time interaction graph**: nodes are asset-window
   instances; *interaction edges* connect assets that communicated in the same
   window; *temporal edges* connect an asset's consecutive observed windows.
5. Trains a **stage detector**: Random Forest (tabular) + GCN (graph), combined
   by a logistic-regression **stacking meta-learner** trained on out-of-fold,
   asset-disjoint base-learner predictions.
6. Trains an **impact forecaster**: a dual-stream GRU (one stream gated by
   IAD evidence, one by LMEP evidence) that predicts whether an asset reaches
   IMP within a horizon H, using **only pre-impact evidence** (IMP is
   structurally unreachable by the input pipeline — see
   `src/training/sequence_builder.py`).
7. Evaluates both stages (Precision/Recall/F1/Macro-F1/per-class F1 + confusion
   matrix for stage detection; ROC-AUC/PR-AUC/F1/Capture@{1,2,5,10}% for
   forecasting) with bootstrap confidence intervals, and saves all metrics to
   CSV/JSON plus figures (confusion matrices, Macro-F1 comparison, PR curves,
   sensitivity plots).
8. Runs the reviewer-response experiments: window-size sensitivity, horizon
   sensitivity, graph ablation, meta-learner ablation, an RF baseline-tuning
   table (selected on validation only), and an inference-latency table.

**Non-negotiable rules enforced throughout the codebase** (see `src/preprocessing/
splitting.py`, `src/preprocessing/features.py`, `src/training/sequence_builder.py`):
asset-disjoint splitting everywhere an asset identity exists (never a random
record split); all preprocessing/scaling/graph-construction/threshold decisions
fit on TRAIN only; the impact forecaster never sees IMP-stage evidence.

---

## 1. Setup

```bash
cd iiot-gnn-ids
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Requires Python 3.10+. Tested with `torch==2.13 (cpu)` and `torch_geometric==2.8`.

## 2. Get the data

This repo does **not** bundle the real datasets (license + size). Download them and
place the CSV(s) here:

| Dataset | Expected file | Place in |
|---|---|---|
| ToN-IoT (Network) | `Train_Test_Network.csv` | `data/raw/toniot/` |
| Edge-IIoTset | `ML-EdgeIIoT-dataset.csv` | `data/raw/edgeiiotset/` |

- ToN-IoT: https://research.unsw.edu.au/projects/toniot-datasets
- Edge-IIoTset: Ferrag et al., IEEE Access 2022 (dataset hosted via IEEE
  DataPort / Kaggle mirrors — search "Edge-IIoTset").

If the exact filename above isn't found, the adapter falls back to globbing
`data/raw/<dataset>/*.csv`, so multiple CSV shards are fine too.

**No real data yet / just want to see the pipeline run?** Generate small synthetic
CSVs that mimic each dataset's real schema:

```bash
python scripts/generate_synthetic_data.py --n-assets 40 --hours 6 --rate-hz 0.05
```

⚠️ **Synthetic data is for smoke-testing the code only.** Results produced on it
have no scientific meaning — every script that runs on synthetic data will still
happily produce metrics/figures, but they describe the synthetic generator, not
either real dataset. Delete `data/raw/*/*.csv` and drop in the real files before
drawing any conclusion.

## 3. Run

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

# Impact forecasting (dual-stream GRU). For Edge-IIoTset this first checks a
# data-quality gate (asset cardinality, timestamp parseability) and WRITES A
# REPORT explaining why it was skipped if the gate fails, instead of
# producing untrustworthy numbers -- see results/edgeiiotset/impact_forecasting/.
python scripts/run_impact_forecasting.py --dataset toniot
python scripts/run_impact_forecasting.py --dataset edgeiiotset

# Reviewer-response experiments (window/horizon sensitivity, graph ablation,
# meta-learner ablation, RF tuning table, inference latency table)
python scripts/run_sensitivity.py --dataset toniot
python scripts/run_sensitivity.py --dataset edgeiiotset
```

Useful flags (all scripts): `--graph-mode {interaction_only,temporal_only,both}`,
`--meta-learner {logistic_regression,mlp,gradient_boosting}`, `--n-folds`,
`--delta-t` (window size override), `--seed`. Run `--help` on any script for the
full list.

## 4. Where results go

```
results/
  toniot/
    stage_detection/
      stage_metrics.json          # per-class P/R/F1, Macro-F1, confusion matrix, bootstrap CI
      stage_per_class_metrics.csv
      stage_summary.csv
      confusion_matrix_stacked_test.png
      macro_f1_comparison.png
    impact_forecasting/
      forecast_metrics.json       # ROC-AUC, PR-AUC, F1, Capture@{1,2,5,10}%, bootstrap CI
      forecast_summary.csv
      forecast_predictions.csv    # per-instance (asset, cut window, y_true, y_proba)
      pr_curve_test.png
    sensitivity/
      window_sensitivity.{csv,png}
      horizon_sensitivity.{csv,png}
      graph_ablation.{csv,png}
      meta_learner_ablation.{csv,png}
      baseline_rf_tuning.csv
      inference_latency.csv
  edgeiiotset/
    ... same layout ...
    impact_forecasting/SKIPPED_report.json   # only if the data-quality gate fails
```

## 5. Configuration

Everything data- or model-specific lives in YAML, not in code:

- `configs/toniot.yaml`, `configs/edgeiiotset.yaml` — dataset schema candidates
  (columns the adapter looks for), window size, forecasting horizon, split
  fractions, graph construction, and every model hyperparameter.
- `configs/stage_mapping_toniot.yaml`, `configs/stage_mapping_edgeiiotset.yaml` —
  raw attack-type → {Benign, IAD, LMEP, IMP} mapping, with the rationale for each
  mapping documented inline.

To change a hyperparameter, edit the relevant YAML — no code changes needed.
`src/utils/config.override()` is used internally (e.g. by `run_sensitivity.py`'s
window-size sweep) to produce a modified in-memory copy without touching the file.

## 6. Design notes / why things are structured this way

- **Dataset adapters never invent columns.** `src/data/generic_adapter.py`
  inspects the real CSV header at load time and matches it against the
  candidate names in each dataset's YAML config; a required-but-missing field
  raises a clear error instead of guessing.
- **Asset-disjoint splitting everywhere.** `src/preprocessing/splitting.py`
  splits by *unique asset*, not by record — the same rule is re-applied for
  every cross-validation fold used in out-of-fold stacking
  (`src/training/stage_detector_trainer.py`, via `sklearn.GroupKFold`).
- **No test-time information ever touches training.** Feature scaling/encoding
  (`src/preprocessing/features.py`) fits only on the TRAIN split; the RF
  hyperparameter table in `run_sensitivity.py` is selected on VAL only; the GCN
  used to generate out-of-fold stacking features is trained on a subgraph
  induced over TRAIN nodes ONLY (`AssetTimeGraph.induced_subgraph`), so val/test
  node *features* — not just their labels — are structurally unreachable during
  that phase.
- **No IMP leakage into the forecaster.** `src/training/sequence_builder.py`
  guarantees no window at or after an asset's first IMP-labeled window can ever
  appear in a forecasting input sequence; `src/training/impact_forecast_trainer.py`
  additionally drops the IMP probability dimension entirely from the stage-
  evidence features it does use.
- **Edge-IIoTset's impact-forecasting validity is checked, not assumed.**
  `src/data/edgeiiotset_adapter.py` gates on asset cardinality and timestamp
  parseability; `scripts/run_impact_forecasting.py` reports and skips (rather
  than silently producing numbers) if the gate fails.

## 7. Project structure

```
src/
  data/            dataset schema, base + per-dataset adapters, loader factory
  preprocessing/   asset-disjoint split, entity-time-window builder, feature scaling
  mapping/         raw attack-type -> {Benign, IAD, LMEP, IMP} mapping
  graph/           asset-time interaction graph construction (+ ablation modes)
  models/          RandomForest wrapper, GCN, dual-stream GRU, meta-learner
  training/        GCN trainer, stage-detector (RF+GCN+stacking) trainer,
                    forecasting sequence builder + trainer
  evaluation/       metrics (stage + forecasting), bootstrap CIs, plotting, palette
  utils/            config loading, logging, seeding
  pipeline.py       shared load->map->split->window->graph orchestration
configs/
  toniot.yaml, edgeiiotset.yaml                       dataset schema + hyperparameters
  stage_mapping_toniot.yaml, stage_mapping_edgeiiotset.yaml
scripts/
  generate_synthetic_data.py   smoke-test data generator (NOT real data)
  run_stage_detection.py
  run_impact_forecasting.py
  run_sensitivity.py
  run_all.py
results/            all metrics/CSVs/figures land here, per dataset
```
