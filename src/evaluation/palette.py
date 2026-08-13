"""Shared color palette for all matplotlib figures produced by this project
(confusion matrices, Macro-F1 comparisons, PR curves, sensitivity plots).

Values are the validated reference palette from the project's dataviz
skill (categorical hues chosen for maximum adjacent-pair colorblind-safe
separation; sequential ramp for ordinal/magnitude encodings). Kept in one
place so every figure reads as one consistent system.
"""
from __future__ import annotations

# Fixed-order categorical hues (never cycle/reassign based on data — always
# index into this list in the same order for the same series identity).
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#1baf7a",  # 2 aqua
    "#eda100",  # 3 yellow  (sub-3:1 contrast on light bg -> pair with direct labels)
    "#008300",  # 4 green
    "#4a3aa7",  # 5 violet
    "#e34948",  # 6 red
    "#e87ba4",  # 7 magenta
    "#eb6834",  # 8 orange
]

# Sequential single-hue ramp (blue), light -> dark, for ordinal/magnitude
# encodings (e.g. the 4-stage Benign->IAD->LMEP->IMP confusion matrix).
SEQUENTIAL_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# Fixed model-identity -> color mapping used across every figure in this
# project, so "RF" is always the same hue whichever plot you're looking at.
MODEL_COLORS = {
    "RF": CATEGORICAL[0],
    "GCN": CATEGORICAL[1],
    "Stacked": CATEGORICAL[4],  # violet, not slot 3 (yellow) -> keeps contrast clean without relying on labels
}

STAGE_COLORS = {
    "Benign": SEQUENTIAL_BLUE[0],
    "IAD": SEQUENTIAL_BLUE[2],
    "LMEP": SEQUENTIAL_BLUE[4],
    "IMP": "#0d366b",
}
