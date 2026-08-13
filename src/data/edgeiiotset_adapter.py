"""Edge-IIoTset dataset adapter.

Edge-IIoTset's `frame.time` is nominally a Wireshark-style timestamp string
and it does expose `ip.src_host` / `ip.dst_host`, so per the project spec we
attempt the FULL pipeline (stage detection + graph + impact forecasting) on
it. However the publicly distributed `ML-EdgeIIoT-dataset.csv` has a
documented export defect verified at load time (see `parse_timestamp`
below): `frame.time` is missing its calendar-date component (month/day) for
100% of rows with a real value. `pandas.to_datetime`'s dateutil fallback
does NOT raise/NaT on a date-less string like " 2021 22:14:30.939803000" —
it silently defaults the missing month/day to January 1st. Trusting that
blindly would make every row across ~13 real distinct capture sessions
collapse onto one fabricated calendar date, corrupting any real Delta-t
time-window bucketing and any "consecutive window" temporal-edge/forecast-
sequence construction, while looking superficially "99% parseable".

Design: we do NOT drop rows over this (that would also discard the 1,214
real, validly-labeled MITM rows, whose `frame.time` is a different
placeholder ("0.0"/"6.0") -- and record-level fields other than time are
completely real and usable). Instead:
  - `parse_timestamp` still returns pandas' best-effort parse (Jan-1
    default included) so every row keeps SOME datetime value and no row is
    dropped for "unparseable" timestamp.
  - Separately, we compute the TRUE fraction of rows whose raw string
    contains an actual calendar date (month name or D/M/Y-style numeric
    date), and gate `timestamp_reliable_for_windowing` on THAT, not on
    pandas' naive parseability.
  - When `timestamp_reliable_for_windowing` is False, `src/pipeline.py`
    falls back to RECORD-LEVEL (not asset-TIME-window, and not one-node-
    per-asset either -- see build_record_level_instances' docstring for why
    asset-level aggregation was rejected: it collapses ~99.9% of assets to
    the worst-ever IMP class) instances, with an interaction graph built
    from real observed src/dst communication pairs (window/time-
    independent) and temporal edges explicitly discarded (no trustworthy
    order to derive them from). Impact forecasting, window-size
    sensitivity, and horizon sensitivity all require a real per-asset
    sequence of multiple time-ordered windows, so those remain gated off
    and are written up as SKIPPED_WITH_REASON by the scripts that use this
    quality flag.
"""
from __future__ import annotations

import re

import pandas as pd

from src.data.generic_adapter import GenericAdapter
from src.data.schema import DatasetBundle
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_MONTH_RE = re.compile(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", re.IGNORECASE)
_DATE_NUMERIC_RE = re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")


class EdgeIIoTAdapter(GenericAdapter):
    def __init__(self, config):
        super().__init__(config)
        self._calendar_date_frac: float = 1.0  # updated by parse_timestamp

    def parse_timestamp(self, raw: pd.Series, fmt: str) -> pd.Series:
        raw_str = raw.astype(str)
        has_real_date = raw_str.str.contains(_MONTH_RE) | raw_str.str.contains(_DATE_NUMERIC_RE)
        self._calendar_date_frac = float(has_real_date.mean()) if len(has_real_date) else 0.0
        # Best-effort parse for bookkeeping only -- NOT trusted for chronological
        # windowing unless extra_quality_checks confirms timestamp_reliable_for_windowing.
        return pd.to_datetime(raw_str, errors="coerce")

    def extra_quality_checks(self, bundle: DatasetBundle) -> None:
        fc_cfg = self.config.get("forecasting", {})
        min_assets = fc_cfg.get("min_distinct_assets", 5)
        min_calendar_date_frac = fc_cfg.get("min_parseable_timestamp_frac", 0.95)

        n_assets = bundle.quality["n_distinct_assets"]
        calendar_date_frac = self._calendar_date_frac
        bundle.quality["timestamp_parseable_frac_naive"] = bundle.quality["timestamp_parseable_frac"]
        bundle.quality["timestamp_calendar_date_frac"] = calendar_date_frac
        timestamp_reliable_for_windowing = calendar_date_frac >= min_calendar_date_frac
        bundle.quality["timestamp_reliable_for_windowing"] = timestamp_reliable_for_windowing

        reasons = []
        if n_assets < min_assets:
            reasons.append(
                f"only {n_assets} distinct assets found (< required {min_assets}); "
                f"too few independent units for an asset-disjoint split to be meaningful."
            )
        if not timestamp_reliable_for_windowing:
            reasons.append(
                f"only {calendar_date_frac:.1%} of rows have a genuine calendar date "
                f"(month/day) in `frame.time` (< required {min_calendar_date_frac:.1%}) -- "
                f"the published ML-EdgeIIoT-dataset.csv is missing month/day for essentially "
                f"every row (naive pandas parsing silently defaults it to Jan-1, which is NOT "
                f"a real timestamp); real chronological Delta-t windowing, temporal graph "
                f"edges, and pre-impact sequence construction are not valid on this field."
            )
        if not bundle.has_peer_id:
            reasons.append("no destination/peer-id column found, so asset interaction cannot be established.")

        bundle.quality["impact_forecasting_valid"] = len(reasons) == 0
        bundle.quality["impact_forecasting_invalid_reasons"] = reasons
        bundle.quality["window_sensitivity_valid"] = timestamp_reliable_for_windowing
        bundle.quality["window_sensitivity_invalid_reason"] = (
            "" if timestamp_reliable_for_windowing else
            f"frame.time lacks a real calendar date for {(1 - calendar_date_frac):.1%} of rows; "
            f"sweeping Delta-t is meaningless without trustworthy absolute time."
        )
        if reasons:
            msg = "Impact forecasting will be SKIPPED for edgeiiotset: " + "; ".join(reasons)
            bundle.warnings.append(msg)
        if not timestamp_reliable_for_windowing:
            bundle.warnings.append(
                "Falling back to RECORD-LEVEL (non-windowed) instances for edgeiiotset stage "
                "detection + graph construction: one node per raw record (asset-level "
                "aggregation was rejected -- it collapses ~99.9% of assets to the worst-ever "
                "IMP class on this dataset), interaction edges built from real observed "
                "src/dst pairs (window/time-independent, via one representative node per "
                "asset), temporal edges discarded (no trustworthy chronological order). "
                "See timestamp_calendar_date_frac."
            )
