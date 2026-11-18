#!/usr/bin/env python3
"""Figure: the 19 suggested designs of the independent test (group C) against the 2-point LEED target.

Each design is drawn at its predicted position (hollow) and its simulated position (filled), joined by a
thin line; the shaded area is the target (sDA >= 55 % and ASE <= 10 %). Reads
thesis_data/results/extensions/independent_validation/predicted_vs_radiance_147_designs.csv; no new simulation.

    python thesis_data/figure_scripts/fig_independent_groupC.py

Output: thesis_data/results/extensions/independent_validation/fig_independent_validation_groupC.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "thesis_data" / "results" / "extensions" / "independent_validation"
SDA, ASE = "sDA_300_50_percent", "ASE_1000_250_percent"
SDA_MIN, ASE_MAX = 55.0, 10.0
BLUE, RED = "#2a78d6", "#d03b3b"
INK, INK2, MUTED, GRID, TARGET = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#e8f1fb"
LABEL_OFFSET = {"C011": (-7, -14), "C016": (-9, 0),"C018": (-7, 9), "C014": (9, 0)}   # points
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
})


def main():
    d = pd.read_csv(OUT / "predicted_vs_radiance_147_designs.csv")
    c = d[d["group"] == "C"].copy()
    c["reached"] = (c[f"sim_{SDA}"] >= SDA_MIN) & (c[f"sim_{ASE}"] <= ASE_MAX)
    assert len(c) == 19 and c["reached"].sum() == 15

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    xlim, ylim = (51, 72), (7.5, 10.8)
    ax.axvspan(SDA_MIN, xlim[1], ymin=0, ymax=(ASE_MAX - ylim[0]) / (ylim[1] - ylim[0]), color=TARGET, linewidth=0, zorder=0)
    ax.axvline(SDA_MIN, color=INK2, linewidth=1.0, zorder=1)
    ax.axhline(ASE_MAX, color=INK2, linewidth=1.0, zorder=1)

    for reached, color, marker, n_label in [(True, BLUE, "o", "Reached the target"), (False, RED, "X", "Missed the target")]:
        g = c[c["reached"] == reached]
        for _, r in g.iterrows():
            ax.plot([r[f"pred_{SDA}"], r[f"sim_{SDA}"]], [r[f"pred_{ASE}"], r[f"sim_{ASE}"]],
                    color=color, linewidth=0.8, alpha=0.55, zorder=2)
        ax.scatter(g[f"pred_{SDA}"], g[f"pred_{ASE}"], s=30, facecolors="white", edgecolors=color, linewidths=1.0, zorder=3)
        ax.scatter(g[f"sim_{SDA}"], g[f"sim_{ASE}"], s=62 if reached else 90, color=color, marker=marker,
                   edgecolors="white", linewidths=0.8, zorder=4, label=f"{n_label} in Radiance ({len(g)})")
    ax.scatter([], [], s=30, facecolors="white", edgecolors=INK2, linewidths=1.0, label="Suggestion (MLP prediction)")

    for _, r in c[~c["reached"]].iterrows():
        ax.annotate(r["id"], (r[f"sim_{SDA}"], r[f"sim_{ASE}"]), xytext=LABEL_OFFSET[r["id"]], textcoords="offset points",
                    fontsize=10, weight="bold", color=INK, ha="right" if r["id"] == "C016" else "left", va="center", zorder=5)
    ax.text(xlim[1] - 0.25, ASE_MAX - 0.07, "Target: 2 LEED points\nsDA ≥ 55 %, ASE ≤ 10 %", ha="right", va="top", fontsize=10, color=INK2)

    ax.set(xlim=xlim, ylim=ylim, xlabel="sDA300/50% [%]", ylabel="ASE1000,250h [%]")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3,
              handletextpad=0.3, columnspacing=1.0)
    fig.savefig(OUT / "fig_independent_validation_groupC.png", dpi=300, bbox_inches="tight")
    print(OUT / "fig_independent_validation_groupC.png")


if __name__ == "__main__":
    main()
