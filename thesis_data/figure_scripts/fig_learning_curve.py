#!/usr/bin/env python3
"""Figure: learning curves of the three models (thesis Fig. 4.17; reads results of learning_curve.py, no retraining).

Test RMSE on the 112 test cases against the number of training cases, for MLP, XGBoost and linear
regression. Line = mean of the five repeats, band = range (min-max). Linear y axis, the same 0-16 pp
scale in both panels so that the errors of sDA and ASE can be compared directly.

    python thesis_data/figure_scripts/fig_learning_curve.py

Output: thesis_data/results/extensions/learning_curve/fig_learning_curve.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

OUT = config.RESULTS / "extensions" / "learning_curve"
TARGETS = {"sDA_300_50_percent": "sDA300/50%", "ASE_1000_250_percent": "ASE1000,250h"}
MODELS = {"MLP": "#1baf7a", "XGBoost": "#eb6834", "Linear regression": "#2a78d6"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 9.5, "axes.titlecolor": INK, "axes.linewidth": 0.8,
})


def draw(summary, path):
    sizes = sorted(summary.n_train.unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for ax, (t, label) in zip(axes, TARGETS.items()):
        for name, color in MODELS.items():
            s = summary[(summary.model == name) & (summary.target == t)].sort_values("n_train")
            ax.fill_between(s.n_train, s.RMSE_min, s.RMSE_max, color=color, alpha=0.18, linewidth=0)
            ax.plot(s.n_train, s.RMSE_mean, color=color, linewidth=1.8, marker="o", markersize=4, label=name)
        ax.set_xticks(sizes)
        ax.set_xlabel("training cases (Radiance simulations)")
        ax.set_title(label, loc="left")
        ax.set_ylim(0, 16)
        ax.set_yticks(range(0, 17, 2))
        ax.tick_params(axis="y", labelleft=True)          # repeat the tick labels on the right panel
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("test RMSE [pp]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(w_pad=2.0)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    summary = pd.read_csv(OUT / "learning_curve_summary.csv")
    draw(summary, OUT / "fig_learning_curve.png")
    print(f"Saved to {OUT / 'fig_learning_curve.png'}")


if __name__ == "__main__":
    main()
