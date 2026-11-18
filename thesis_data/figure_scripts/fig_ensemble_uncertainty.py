#!/usr/bin/env python3
"""Error margins and ensemble spread on the 112 test cases (thesis Section 4.4.2).

Reads the output of train_mlp_ensemble.py and writes
  fig_mlp_ensemble_error_margin.png   Fig. 4.28 (and error_margins.csv, Table 4.18)
  fig_mlp_ensemble_spread_groups.png  Fig. 4.29: mean absolute error in three equal-size spread groups
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "3_ml"))
import common  # noqa: E402
from fig_mlp_limitations import BLUE, INK, INK2, MUTED, style

ROOT = common.OUT_ROOT / "reduced" / "mlp_ensemble"
FIG = common.OUT_ROOT / "reduced" / "figures"


def main():
    d = pd.read_csv(ROOT / "ensemble_test_predictions.csv", dtype={"model_id": str})
    FIG.mkdir(parents=True, exist_ok=True)
    figure_spread_groups(d)
    figure_error_margin(d)


def figure_spread_groups(d):
    """Test cases split into three equal-size groups by spread; mean absolute error per group."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    for ax, t, tag in zip(axes, common.TARGETS, ("(a)", "(b)")):
        spread = d[f"std_{t}"]
        err = (d[f"mean_{t}"] - d[f"actual_{t}"]).abs()
        group = pd.qcut(spread.rank(method="first"), 3, labels=False)   # rank: ties at spread = 0
        labels, mae = [], []
        for g, name in enumerate(("low", "medium", "high")):
            m = group == g
            labels.append(f"{name} spread\n{spread[m].min():.2f}–{spread[m].max():.2f} pp\nn = {m.sum()}")
            mae.append(err[m].mean())
        bars = ax.bar(range(3), mae, width=0.55, color=BLUE, zorder=2)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in mae], padding=3, fontsize=8, color=INK)
        ax.set_xticks(range(3), labels, fontsize=7.5)
        ax.set_title(f"{tag} {common.TARGET_LABELS[t]}", loc="left")
        style(ax)
        ax.grid(False, axis="x")
    axes[0].set_ylabel("mean absolute error of ensemble mean [pp]")
    axes[0].set_ylim(0, max(axes[0].get_ylim()[1], 1.6))
    fig.tight_layout(w_pad=2.0)
    out = FIG / "fig_mlp_ensemble_spread_groups.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


LEVELS = (0.80, 0.90, 0.95)
LEED = {"sDA_300_50_percent": (55.0, 75.0), "ASE_1000_250_percent": (10.0,)}


def figure_error_margin(d):
    """Cumulative share of test cases versus absolute error of the ensemble mean, with the
    80/90/95% empirical error margins; also writes the margins and how many test cases each
    level would send to Radiance (prediction +/- margin crossing a LEED threshold)."""
    err = {t: (d[f"mean_{t}"] - d[f"actual_{t}"]).abs().to_numpy() for t in common.TARGETS}
    rows = []
    for q in LEVELS:
        m = {t: float(np.quantile(err[t], q)) for t in common.TARGETS}
        near = {t: np.zeros(len(d), bool) for t in common.TARGETS}
        for t in common.TARGETS:
            for thr in LEED[t]:
                near[t] |= (d[f"mean_{t}"] - thr).abs().to_numpy() < m[t]
        rows.append({"level": q, "sDA_margin_pp": m[common.TARGETS[0]], "ASE_margin_pp": m[common.TARGETS[1]],
                     "cases_beyond_margin": int(round(len(d) * (1 - q))),
                     "flagged_sDA": int(near[common.TARGETS[0]].sum()),
                     "flagged_ASE": int(near[common.TARGETS[1]].sum()),
                     "flagged_total": int((near[common.TARGETS[0]] | near[common.TARGETS[1]]).sum())})
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "error_margins.csv", index=False, float_format="%.3f")
    print(table.round(2).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    for ax, t, tag in zip(axes, common.TARGETS, ("(a)", "(b)")):
        e = np.sort(err[t])
        share = np.arange(1, len(e) + 1) / len(e) * 100
        ax.step(e, share, where="post", color=BLUE, linewidth=1.6, zorder=2)
        for q in LEVELS:
            m = np.quantile(e, q)
            main = q == 0.90
            ax.plot([0, m], [q * 100, q * 100], color=INK2 if main else MUTED, linewidth=0.8,
                    linestyle="-" if main else (0, (3, 3)), zorder=1)
            ax.plot([m, m], [0, q * 100], color=INK2 if main else MUTED, linewidth=0.8,
                    linestyle="-" if main else (0, (3, 3)), zorder=1)
            ax.scatter([m], [q * 100], s=22, color=INK if main else MUTED, zorder=3)
            ax.text(m + 0.12, q * 100 - 1.5, f"{q:.0%}: ±{m:.1f} pp", fontsize=7.5, va="top",
                    color=INK if main else INK2, fontweight="bold" if main else "normal", zorder=4,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.8))   # masks the reference lines
        ax.set_xlim(0, e.max() * 1.05)
        ax.set_ylim(0, 102)
        ax.set_xlabel("absolute error of ensemble mean [pp]")
        ax.set_title(f"{tag} {common.TARGET_LABELS[t]}", loc="left")
        style(ax)
    axes[0].set_ylabel("cumulative share of test cases [%]")
    fig.tight_layout(w_pad=2.0)
    out = FIG / "fig_mlp_ensemble_error_margin.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
