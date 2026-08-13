"""Generic pandas.DataFrame -> manuscript-ready LaTeX table writer, used by
every table under results/manuscript_tables/. Kept dependency-free (no
`df.to_latex` / no jinja) so numeric formatting and the SKIPPED/invalid-row
convention are consistent across every table in the project."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


def _fmt_cell(v) -> str:
    if isinstance(v, str):
        return v.replace("_", r"\_")
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, float):
        if np.isnan(v):
            return "--"
        if abs(v) >= 1000:
            return f"{v:,.1f}"
        return f"{v:.3f}"
    if v is None:
        return "--"
    return str(v).replace("_", r"\_")


def df_to_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    save_path: str | Path,
    columns: Optional[Sequence[str]] = None,
    column_headers: Optional[Sequence[str]] = None,
    note: Optional[str] = None,
    float_format: str = "%.3f",
) -> None:
    """Writes a standalone `table` environment (booktabs style) to
    `save_path`. `columns`: optional column subset/order (defaults to all of
    `df`'s columns). `column_headers`: optional display names (must match
    length of `columns`, defaults to the column names with `_` escaped).
    `note`: optional footnote line (e.g. threshold-selection rule, or a
    SKIPPED/validity caveat) rendered below the table via a `\\parbox`.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(columns) if columns is not None else list(df.columns)
    headers = list(column_headers) if column_headers is not None else [c.replace("_", r"\_") for c in cols]
    assert len(headers) == len(cols), "column_headers must match columns length"

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    col_spec = "l" * len(cols)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        cells = [_fmt_cell(row[c]) if c in row.index else "--" for c in cols]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if note:
        escaped_note = note.replace("_", r"\_").replace("%", r"\%")
        lines.append(r"\vspace{2pt}")
        lines.append(f"\\parbox{{\\linewidth}}{{\\footnotesize {escaped_note}}}")
    lines.append(r"\end{table}")

    save_path.write_text("\n".join(lines) + "\n")


def write_skipped_latex_note(
    caption: str, label: str, save_path: str | Path, reasons: List[str], dataset: str,
) -> None:
    """For an experiment written up as SKIPPED_WITH_REASON: still emit a
    small manuscript-ready .tex note (not a data table) explaining why, so
    the LaTeX source has a citable placeholder instead of a missing file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    reason_text = " ".join(reasons).replace("_", r"\_").replace("%", r"\%")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{p{0.9\linewidth}}",
        r"\toprule",
        f"\\textbf{{SKIPPED for {dataset}}} \\\\",
        r"\midrule",
        f"{reason_text} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    save_path.write_text("\n".join(lines) + "\n")
