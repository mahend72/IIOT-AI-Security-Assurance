"""Matplotlib figures for stage detection / impact forecasting results.

Styling follows the project's shared palette (src/evaluation/palette.py):
fixed-order categorical hues for model/series identity, a single-hue
sequential ramp for the ordinal 4-stage confusion matrix, recessive
gridlines, and direct value labels wherever a color's own contrast is
borderline (the yellow categorical slot in particular).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: this project only ever renders to file
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve

from src.evaluation import palette as pal

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.edgecolor": pal.BASELINE,
        "axes.labelcolor": pal.INK_SECONDARY,
        "text.color": pal.INK_PRIMARY,
        "xtick.color": pal.INK_MUTED,
        "ytick.color": pal.INK_MUTED,
        "axes.grid": True,
        "grid.color": pal.GRIDLINE,
        "grid.linewidth": 0.8,
        "figure.facecolor": pal.SURFACE,
        "axes.facecolor": pal.SURFACE,
        "savefig.facecolor": pal.SURFACE,
    }
)

_SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", pal.SEQUENTIAL_BLUE)


def _save(fig, save_path: str | Path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, save_path: str | Path) -> None:
    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_norm, cmap=_SEQ_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, color=pal.INK_PRIMARY, fontsize=12, pad=12)
    ax.grid(False)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = pal.INK_PRIMARY if cm_norm[i, j] < 0.6 else "white"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.0%})", ha="center", va="center", color=text_color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized fraction", color=pal.INK_SECONDARY)
    _save(fig, save_path)


def plot_macro_f1_comparison(
    model_scores: Dict[str, float],
    title: str,
    save_path: str | Path,
    ci: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    names = list(model_scores.keys())
    values = [model_scores[n] for n in names]
    colors = [pal.MODEL_COLORS.get(n, pal.CATEGORICAL[i % len(pal.CATEGORICAL)]) for i, n in enumerate(names)]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    yerr = None
    if ci:
        lower = [values[i] - ci[n][0] for i, n in enumerate(names)]
        upper = [ci[n][1] - values[i] for i, n in enumerate(names)]
        yerr = [lower, upper]
    bars = ax.bar(names, values, color=colors, width=0.55, yerr=yerr, capsize=4, ecolor=pal.INK_MUTED)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.0)
    ax.set_title(title, color=pal.INK_PRIMARY, fontsize=12, pad=12)
    # Direct labels above every bar -- relief rule for the borderline-contrast slots.
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom",
                 color=pal.INK_PRIMARY, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, save_path)


def plot_pr_curves(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str,
    save_path: str | Path,
) -> None:
    """`curves`: {series_name: (y_true, y_score)}."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, (name, (y_true, y_score)) in enumerate(curves.items()):
        if len(y_true) == 0 or len(set(np.asarray(y_true).tolist())) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        color = pal.CATEGORICAL[i % len(pal.CATEGORICAL)]
        ax.plot(recall, precision, color=color, linewidth=2, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title, color=pal.INK_PRIMARY, fontsize=12, pad=12)
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, save_path)


def plot_sensitivity_line(
    x_values: Sequence,
    series: Dict[str, Sequence[float]],
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str | Path,
    x_is_categorical: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x_pos = range(len(x_values)) if x_is_categorical else x_values
    for i, (name, y_values) in enumerate(series.items()):
        color = pal.MODEL_COLORS.get(name, pal.CATEGORICAL[i % len(pal.CATEGORICAL)])
        ax.plot(x_pos, y_values, marker="o", markersize=5, color=color, linewidth=2, label=name)
    if x_is_categorical:
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels([str(x) for x in x_values])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=pal.INK_PRIMARY, fontsize=12, pad=12)
    if len(series) > 1:
        ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, save_path)
