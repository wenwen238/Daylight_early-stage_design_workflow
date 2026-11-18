#!/usr/bin/env python3
"""Figure: predicted vs. simulated sDA and ASE on the 112 test cases, single MLP vs. five-member ensemble.

Single MLP = member 1 of the ensemble (seed 20260924, the model evaluated in Section 4.3.4);
ensemble = mean of the five clipped members (seeds 20260924-20260928). Same layout and style as the
model-comparison figure (fig_model_comparison.py). Reads
thesis_data/results/ml_training/reduced/mlp_ensemble/ensemble_test_predictions.csv; no retraining.

    python thesis_data/figure_scripts/fig_single_vs_ensemble.py

Output: thesis_data/results/extensions/single_vs_ensemble/fig_single_vs_ensemble.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "3_ml"))
import config  # noqa: E402
import common  # noqa: E402

OUT = config.RESULTS / "extensions" / "single_vs_ensemble"
SEED = 20260924
COLUMNS = [("single", "Single MLP (seed 20260924)", "#1baf7a"),
           ("ensemble", "Ensemble of five MLPs", "#8a5aab")]
AXIS_MAX = {"sDA_300_50_percent": 100, "ASE_1000_250_percent": 90}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 9.5, "axes.titlecolor": INK, "axes.linewidth": 0.8,
})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(config.ML_TRAINING / "reduced" / "mlp_ensemble" / "ensemble_test_predictions.csv")
    fig, axes = plt.subplots(2, 2, figsize=(5.2, 5.2), sharex="row", sharey="row")
    rows = []
    for col, (key, label, color) in enumerate(COLUMNS):
        for row, t in enumerate(common.TARGETS):
            ax = axes[row, col]
            hi = AXIS_MAX[t]
            y = d[f"actual_{t}"]
            p = d[f"pred_{t}_seed{SEED}"] if key == "single" else d[f"mean_{t}"]
            mae, rmse, r2 = mean_absolute_error(y, p), float(np.sqrt(((p - y) ** 2).mean())), r2_score(y, p)
            rows.append({"model": key, "target": t, "MAE_pp": mae, "RMSE_pp": rmse, "R2": r2})
            ax.plot([0, hi], [0, hi], color=INK2, linewidth=0.9, zorder=1)
            ax.scatter(y, p, s=14, color=color, alpha=0.75, linewidths=0.4, edgecolors="white", zorder=2)
            ax.text(0.04, 0.96, f"R² = {r2:.3f}\nMAE = {mae:.2f} pp\nRMSE = {rmse:.2f} pp", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8, color=INK)
            ax.set_xlim(-2, hi + 2)
            ax.set_ylim(-2, hi + 2)
            ax.set_aspect("equal")
            ax.grid(True, color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if row == 0:
                ax.set_title(label, fontsize=9.5, weight="bold", color=INK)
            if col == 0:
                ax.set_ylabel(f"{common.TARGET_LABELS[t]}\npredicted [%]")
            if row == 1:
                ax.set_xlabel("simulated (Radiance) [%]")
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    fig.savefig(OUT / "fig_single_vs_ensemble.png", dpi=300, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(OUT / "single_vs_ensemble_metrics.csv", index=False, float_format="%.4f")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
