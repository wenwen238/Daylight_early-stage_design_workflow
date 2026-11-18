#!/usr/bin/env python3
"""Figure: distribution of sDA300/50% and ASE1000,250h over the 560 simulated cases.

(a) histogram of sDA with the LEED thresholds 55 % and 75 %, (b) histogram of ASE with the
10 % threshold, (c) sDA against ASE, with the cases that meet both sDA >= 55 % and ASE <= 10 %
highlighted. Reads the dataset only, no recalculation.

    python thesis_data/figure_scripts/fig_data_distribution.py

Output: thesis_data/results/extensions/data_distribution/fig_data_distribution.png and statistics_560_cases.json
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "3_ml"))
import config  # noqa: E402
import common  # noqa: E402

OUT = config.RESULTS / "extensions" / "data_distribution"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 9, "axes.titlecolor": INK, "axes.linewidth": 0.8,
})


def style(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def threshold(ax, x, label):
    ax.axvline(x, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=3)
    ax.text(x, 1.0, label, transform=ax.get_xaxis_transform(), fontsize=7, color=INK2, ha="center", va="bottom")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = common.load_dataset(common.DEFAULT_CASES, common.DEFAULT_RESULTS)
    sda, ase = d["sDA_300_50_percent"].to_numpy(), d["ASE_1000_250_percent"].to_numpy()
    both = (sda >= 55) & (ase <= 10)
    bins = np.arange(0, 105, 5)                       # 5 % bins; 100 % falls in the last one

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    ax = axes[0]
    ax.hist(np.minimum(sda, 99.999), bins=bins, color=BLUE, edgecolor="white", linewidth=1)
    threshold(ax, 55, "55 %")
    threshold(ax, 75, "75 %")
    ax.set_xlabel("sDA300/50% [%]")
    ax.set_ylabel("number of cases")
    ax.set_title("(a) sDA", loc="left", pad=12)

    ax = axes[1]
    ax.hist(np.minimum(ase, 99.999), bins=bins, color=ORANGE, edgecolor="white", linewidth=1)
    threshold(ax, 10, "10 %")
    ax.set_xlabel("ASE1000,250h [%]")
    ax.set_title("(b) ASE", loc="left", pad=12)

    ax = axes[2]
    ax.axvspan(55, 100, ymin=0, ymax=10 / 90, color=GREEN, alpha=0.12, linewidth=0, zorder=0)
    ax.scatter(sda[~both], ase[~both], s=9, color=MUTED, edgecolor="white", linewidth=0.3, zorder=2)
    ax.scatter(sda[both], ase[both], s=16, color=GREEN, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(10, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.axvline(55, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.text(3, 87, f"green: sDA ≥ 55 % and ASE ≤ 10 %\n({both.sum()} of {len(d)} cases)", fontsize=7, color=INK2,
            ha="left", va="top", bbox=dict(facecolor="white", edgecolor="none", pad=1.0), zorder=4)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 90)
    ax.set_xlabel("sDA300/50% [%]")
    ax.set_ylabel("ASE1000,250h [%]")
    ax.set_title("(c) sDA and ASE of each case", loc="left", pad=12)

    for a in axes:
        style(a)
        a.set_xlim(0, 100)
    axes[2].grid(True, axis="x", color=GRID, linewidth=0.6)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT / "fig_data_distribution.png", dpi=300, bbox_inches="tight")

    summary = {"n_cases": len(d),
               "sDA": {"min": sda.min(), "median": float(np.median(sda)), "mean": sda.mean(), "max": sda.max(),
                       "cases_at_100": int((sda >= 99.999).sum()), "below_55": int((sda < 55).sum()),
                       "55_to_75": int(((sda >= 55) & (sda < 75)).sum()), "at_least_75": int((sda >= 75).sum())},
               "ASE": {"min": ase.min(), "median": float(np.median(ase)), "mean": ase.mean(), "max": ase.max(),
                       "cases_at_0": int((ase <= 0).sum()), "at_most_10": int((ase <= 10).sum())},
               "sDA_at_least_55_and_ASE_at_most_10": int(both.sum()),
               "sDA_at_least_75_and_ASE_at_most_10": int(((sda >= 75) & (ase <= 10)).sum())}
    (OUT / "statistics_560_cases.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=float))
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
