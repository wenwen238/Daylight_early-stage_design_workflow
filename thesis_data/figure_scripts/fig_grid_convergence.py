#!/usr/bin/env python3
"""Figure: grid convergence of sDA and ASE (reads results of grid_convergence.py, no recalculation).

Mean absolute difference to the 32 x 32 values of all 560 cases against the number of sensor points,
for cell-centre grids from 4 x 4 to 24 x 24, one panel per metric.

    python thesis_data/figure_scripts/fig_grid_convergence.py

Output: thesis_data/results/extensions/grid_convergence/fig_grid_convergence.png
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

OUT = config.RESULTS / "extensions" / "grid_convergence"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
COLORS = {"sDA": BLUE, "ASE": ORANGE}
LABELS = {"sDA": "sDA300/50%", "ASE": "ASE1000,250h"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 9, "axes.titlecolor": INK, "axes.linewidth": 0.8,
})


def main():
    t = pd.read_csv(OUT / "grid_convergence.csv")
    cells = t[t.points < 1024]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for ax, (metric, c) in zip(axes, COLORS.items()):  # one panel per metric, so the two lines do not cross
        d = cells[cells.metric == metric].sort_values("points")
        ax.plot(d.points, d.MAE_pp, color=c, linewidth=2, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=1, zorder=3)
        for _, r in d.iterrows():
            if r.grid in ("6 x 6", "12 x 12", "20 x 20") or (metric == "sDA" and r.grid == "16 x 16"):
                continue                                      # sDA 16 x 16 would sit on the Radiance label
            ax.annotate(r.grid.replace(" x ", "×"), (r.points, r.MAE_pp), textcoords="offset points",
                        xytext=(4, 6), ha="left", fontsize=7, color=INK2)
        ax.set_title(LABELS[metric], loc="left")
        ax.set_xlim(0, 650)
        ax.set_ylim(0, 4)
        ax.set_xlabel("number of sensor points (∝ simulation time)")
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].axhline(0.8, color=MUTED, linewidth=0.8, linestyle=(0, (1, 2)), zorder=1)
    axes[0].text(645, 0.86, "repeated Radiance runs: ≤ 0.8 pp", fontsize=7, color=MUTED, ha="right",
                 va="bottom", bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    axes[0].set_ylabel("grid error of sDA / ASE\n(compared to the 32 × 32 grid) [pp]")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "fig_grid_convergence.png", dpi=300, bbox_inches="tight")
    print(f"Saved to {OUT / 'fig_grid_convergence.png'}")


if __name__ == "__main__":
    main()
