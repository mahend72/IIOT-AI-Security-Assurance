# INVALID — ToN-IoT results archived pre window-fix

**Archived:** 2026-08-20

## Why these results are invalid

`src/preprocessing/windowing.py::assign_window_id` assumed
`df[TIMESTAMP_COL]` was always pandas `datetime64[ns]`:

```python
epoch_seconds = df[TIMESTAMP_COL].astype("int64") / 1e9  # ns -> s
```

Under the pandas version installed in this environment (3.0.1), which
supports non-nanosecond datetime64 resolutions, `pd.to_datetime(numeric,
unit="s")` in `src/data/generic_adapter.py::parse_timestamp` returns
`datetime64[s]`, not `datetime64[ns]`, for the ToN-IoT `ts` column.
`.astype("int64")` on a `datetime64[s]` series already returns whole
seconds, so dividing by `1e9` again crushed every timestamp in the dataset
(epoch range ~1,554,198,358–1,556,549,129) down to a value between 1.554
and 1.557. `floor(that / 60)` is `0` for every single record.

**Confirmed:** `assign_window_id(df, 60).nunique() == 1` on the real
`data/raw/toniot/Train_Test_Network.csv` before the fix.

## Effect on these results

With `window_id` constant across the whole dataset, `build_asset_window_instances`'s
`groupby(["asset_id", "window_id"])` degenerated to `groupby(["asset_id"])`
— i.e. every "entity-window" instance in these archived results is actually
one row per distinct `src_ip`, aggregated (via `max_severity`) over that
asset's **entire ~27-day observed history**, not over any real 60-second
window. This produced exactly 11,536 window rows (== `nunique(src_ip)`),
with only 3 IAD / 5 LMEP / 11 IMP non-Benign instances surviving (the worst
stage ever seen per asset, across its whole lifetime) — vastly understating
real attack-window volume and invalidating every downstream stage-detection,
graph-construction, and impact-forecasting result for ToN-IoT computed from
these window instances. Everything under `toniot/`, `figures_toniot/`, and
`manuscript_tables_toniot/` in this archive directory was produced under
that bug and **must not be reused or cited**.

Edge-IIoTset results are unaffected — that adapter's `timestamp_reliable_for_windowing`
gate routes it through `build_record_level_instances`, which uses an
explicit `window_id_override` and never calls `assign_window_id`.

## Fix

`assign_window_id` was made resolution-independent (see
`src/preprocessing/windowing.py` and `git log` after this archive date).
Corrected results live back under `results/toniot/`, `results/figures/toniot/`,
and `results/manuscript_tables/tab_toniot_*.tex`.
