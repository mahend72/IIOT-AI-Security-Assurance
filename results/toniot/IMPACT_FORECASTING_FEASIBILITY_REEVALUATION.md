# ToN-IoT Impact-Forecasting Feasibility — Post-Fix Re-Evaluation (Step 5)

Data-only re-evaluation using the corrected `assign_window_id`. **No forecasting
model was trained to produce this report** — it only reconstructs
`build_forecast_instances` output and runs it through the codebase's own
`src/evaluation/forecast_validity.py::check_instance_level_validity` gate,
exactly as `scripts/run_main_results.py::section_impact_forecasting` will do
when Step 6 runs for real.

## Conclusion: still SKIPPED_WITH_REASON, but the reason has changed

The **pre-fix** conclusion ("zero pre-impact forecasting instances could be
constructed at any split") is now **false** — real Delta-t=60s windowing
produces plenty of instances. But the corrected data uncovers a *different*,
genuine feasibility problem:

```
check_instance_level_validity(instances) -> (False, [
  "TEST split has only one class present (0 positive / 798 negative of 798 "
  "instances) -- ROC-AUC/PR-AUC/Capture@k are undefined without both classes."
])
```

`gate: instance-level (post sequence-construction)` — the same gate path
`run_main_results.py` already uses, so Step 6 will correctly auto-skip this
without any special-casing needed.

## Why: ToN-IoT's attack episodes are mostly "instant-onset", not escalating

Recomputing `first_imp_pos` (position of an asset's first IMP-labeled
window) per asset that ever reaches IMP:

| asset_id | split | n_windows | first_imp_pos | pre-impact windows available |
|---|---|---|---|---|
| 13.55.50.68 | test | 7 | 0 | **0** |
| 192.168.1.190 | train | 2,492 | 1,729 | 1,729 |
| 192.168.1.193 | train | 1,868 | 1,722 | 1,722 |
| 192.168.1.30 | val | 77 | 18 | 18 |
| 192.168.1.31 | test | 168 | 12 | 12 |
| 192.168.1.33 | train | 68 | 0 | **0** |
| 192.168.1.34 | train | 45 | 0 | **0** |
| 192.168.1.37 | train | 68 | 0 | **0** |
| 192.168.1.38 | val | 1 | 0 | **0** |
| 192.168.1.39 | train | 15 | 0 | **0** |
| 203.14.129.10 | test | 7 | 0 | **0** |

Of the **11** assets that ever reach IMP (out of 11,536 total assets), **7
have IMP as their very first observed window** — i.e. that source IP's
traffic in this capture *is* the attack burst (dos/ddos/ransomware fired at
high volume for a few seconds to minutes), with no earlier
reconnaissance/lateral-movement (IAD/LMEP) traffic from the *same asset* in
the window sequence to forecast from. `build_forecast_instances` correctly
produces **zero** eligible cut points for these assets (`eligible_positions
= range(first_imp_pos) = range(0)`), per its documented leakage rule (no
cut point may ever come at/after IMP).

Only **4** assets have any real pre-impact history: `192.168.1.190` and
`192.168.1.193` (long-lived, mostly-benign LAN hosts with a late attack
episode — 1,729 / 1,722 pre-impact windows), `192.168.1.30` (18), and
`192.168.1.31` (12). Even among those, the forecasting **horizon**
(`horizon_multiple=30`, i.e. 30 windows = 30 minutes at Delta t=60s) only
labels a cut point positive if its `window_id` gap to the first-IMP window
is `<= 30`. That is satisfied by a handful of windows near the end of
`192.168.1.190` and `192.168.1.193`'s pre-impact runs (both in **train**),
giving the **5** positive instances observed — and evidently *not*
satisfied for `192.168.1.30` (val) or `192.168.1.31` (test), whose nearest
pre-impact windows to their own first-IMP window fall outside the 30-window
horizon, so they contribute negatives only.

## Instance counts (data-only reconstruction)

- horizon_multiple = 30 x Delta t (60s) = 30-minute pre-impact horizon, max_seq_len = 60
- Total valid pre-impact forecasting instances: **21,313**
- Positive instances: **5** (all in `train`)

| split | count | positive |
|---|---|---|
| train | 19,124 | 5 |
| val | 1,391 | 0 |
| test | 798 | 0 |

## Recommendation

**Do not run/report the dual-stream GRU forecaster as a statistically
defensible result on ToN-IoT.** With zero positive instances in both `val`
and `test`, ROC-AUC / PR-AUC / Capture@k are mathematically undefined on
held-out data (`sklearn.metrics.roc_auc_score` would raise on a
single-class `y_true`), and 5 positive instances in `train` (out of 19,124)
is far too few to fit or meaningfully validate a sequence model in any
split. This is not an artifact of the windowing bug — it is a real property
of ToN-IoT's `Train_Test_Network.csv`: attack scenarios were captured as
short, high-intensity, often-instantaneous bursts from a small number of
hosts, giving almost no naturally observable "early warning" window before
impact for a forecaster to learn from at asset-level Delta-t=60s
granularity, and an asset-disjoint split (required to avoid leakage)
concentrates the very few exceptions into a single split by chance.

Step 6 should therefore run `scripts/run_main_results.py --dataset toniot`
(and any other forecasting entry points) as normal and let the
existing `check_instance_level_validity` gate auto-write
`impact_forecasting_SKIPPED_WITH_REASON.json` with this reason — no
special-casing required, since this is exactly the gate the codebase
already has for this situation. Stage detection, graph construction, and
all non-forecasting experiments ARE valid and should proceed.
