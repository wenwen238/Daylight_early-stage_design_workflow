#!/usr/bin/env python3
"""Thesis figure: predicted vs. simulated sDA and ASE on the test set for the three models."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "3_ml"))
import common  # noqa: E402

FEATURE_SET = "reduced"
MODELS = [("linear", "Linear regression", "#2a78d6"),
          ("xgboost", "XGBoost", "#eb6834"),
          ("mlp", "MLP", "#1baf7a")]
AXIS_MAX = {"sDA_300_50_percent": 100, "ASE_1000_250_percent": 90}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 9.5, "axes.titlecolor": INK, "axes.linewidth": 0.8,
})


def main():
    root = common.OUT_ROOT / FEATURE_SET
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.2), sharex="row", sharey="row")
    for col, (key, label, color) in enumerate(MODELS):
        pred = pd.read_csv(root / key / "test_predictions.csv")
        for row, target in enumerate(common.TARGETS):
            ax = axes[row, col]
            hi = AXIS_MAX[target]
            actual, predicted = pred[f"actual_{target}"], pred[f"predicted_{target}"]
            ax.plot([0, hi], [0, hi], color=INK2, linewidth=0.9, zorder=1)
            ax.scatter(actual, predicted, s=14, color=color, alpha=0.75, linewidths=0.4,
                       edgecolors="white", zorder=2)
            ax.set_xlim(-2, hi + 2)
            ax.set_ylim(-2, hi + 2)
            ax.set_aspect("equal")
            ax.grid(True, color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if row == 0:
                ax.set_title(label, loc="center", fontsize=9.5, weight="bold", color=INK)
            if col == 0:
                ax.set_ylabel(f"{common.TARGET_LABELS[target]}\npredicted [%]")
            if row == 1:
                ax.set_xlabel("simulated (Radiance) [%]")
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    path = out / "fig_model_comparison_test.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
