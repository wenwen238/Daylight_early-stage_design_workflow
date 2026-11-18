#!/usr/bin/env python3
"""Limitations of the selected MLP: held-out residuals and physically implausible responses.

Held-out predictions: the 448 training cases are predicted by 5-fold cross-validation
(same folds and selected hyperparameters as train_mlp.py), the 112 test cases by the
saved final model. Writes two thesis figures, the held-out predictions and the
numbers quoted in the text.
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "3_ml"))
import common  # noqa: E402
import train_mlp

ROOT = common.OUT_ROOT / "reduced" / "mlp"
FIG = common.OUT_ROOT / "reduced" / "figures"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#008300"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 9.5, "axes.titlecolor": INK, "axes.linewidth": 0.8,
})


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def heldout_predictions(data, train, test, feats, models):
    best = json.load(open(ROOT / "best_params.json", encoding="utf-8"))
    d = pd.concat([train.assign(split="cv"), test.assign(split="test")], ignore_index=True)
    for t in common.TARGETS:
        params = {k: tuple(v) if isinstance(v, list) else v for k, v in best[t].items()}
        est = train_mlp.make_estimator("reduced").set_params(**params)
        oof = cross_val_predict(est, train[feats], train[t], cv=common.cv_folds(train), n_jobs=5)
        raw = np.concatenate([oof, models[t].predict(test[feats])])
        d[f"raw_{t}"] = raw
        d[f"pred_{t}"] = common.clip(raw)
        d[f"res_{t}"] = d[f"pred_{t}"] - d[t]
    return d


def sweep(model, row, feats, col, grid):
    X = pd.DataFrame([row[feats]] * len(grid))
    X[col] = grid
    return model.predict(X)


def figure_residuals(d):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    bins = {"sDA_300_50_percent": np.arange(0, 101, 10), "ASE_1000_250_percent": np.arange(0, 91, 10)}
    for ax, t, tag in zip(axes, common.TARGETS, ("(a)", "(b)")):
        for wt, color, label in (("conventional", BLUE, "conventional window"),
                                 ("floor_to_ceiling", ORANGE, "floor-to-ceiling window")):
            s = d[d.window_type == wt]
            ax.scatter(s[t], s[f"res_{t}"], s=9, color=color, alpha=0.55, linewidths=0, label=label)
        centres, means, maes = [], [], []
        for lo, hi in zip(bins[t][:-1], bins[t][1:]):
            m = (d[t] >= lo) & (d[t] < hi if hi < bins[t][-1] else d[t] <= hi)
            if m.sum() >= 5:
                centres.append((lo + hi) / 2)
                means.append(d.loc[m, f"res_{t}"].mean())
                maes.append(d.loc[m, f"res_{t}"].abs().mean())
        ax.plot(centres, maes, color=INK2, linewidth=1.4, linestyle=(0, (4, 2)), marker="s", markersize=3.2,
                label="bin MAE")
        ax.plot(centres, means, color=INK, linewidth=1.6, marker="o", markersize=3.5, label="bin mean (bias)")
        ax.axhline(0, color=INK2, linewidth=0.9)
        ax.set_xlabel("simulated value (Radiance) [%]")
        ax.set_ylabel("predicted − simulated [pp]")
        ax.set_title(f"{tag} {common.TARGET_LABELS[t]}", loc="left")
        lim = np.ceil(d[f"res_{t}"].abs().max() + 0.5)
        ax.set_ylim(-lim, lim)
        style(ax)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / "fig_mlp_heldout_residuals.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_north_only(data, feats, models):
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    ase = models["ASE_1000_250_percent"]
    widths = np.linspace(0.6, 2.4, 37)
    north = data[data.active_facades == "north"]
    for _, row in north.iterrows():
        color = BLUE if row.window_type == "conventional" else ORANGE
        ax.plot(widths, common.clip(sweep(ase, row, feats, "window_width_m", widths)), color=color,
                alpha=0.45, linewidth=0.9)
    # the simulated value is a data series, not an axis: coloured and dashed, drawn above the curves
    ax.axhline(0, color=GREEN, linewidth=1.8, linestyle=(0, (5, 3)), zorder=3)
    ax.plot([], [], color=BLUE, label="MLP prediction, conventional window")
    ax.plot([], [], color=ORANGE, label="MLP prediction, floor-to-ceiling window")
    ax.plot([], [], color=GREEN, linewidth=1.8, linestyle=(0, (5, 3)),
            label="simulated value (0 for every north-only case)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_xlabel("window width [m]")
    ax.set_ylabel("ASE1000,250h [%]")
    ax.set_ylim(-0.1, None)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_mlp_north_only_ase.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def numbers(d, data, feats, models):
    out = {}
    for t in common.TARGETS:
        r, a = d[f"res_{t}"], d[t]
        g = {}
        g["overall"] = {"MAE": r.abs().mean(), "bias": r.mean(), "max_abs": r.abs().max()}
        g["clipped_below_0"] = int((d[f"raw_{t}"] < 0).sum())
        g["clipped_above_100"] = int((d[f"raw_{t}"] > 100).sum())
        for wt in ("conventional", "floor_to_ceiling"):
            for ov in (False, True):
                m = (d.window_type == wt) & ((d.overhang_depth_m > 0) == ov)
                g[f"{wt}_overhang_{ov}"] = {"n": int(m.sum()), "MAE": r[m].abs().mean(), "bias": r[m].mean()}
        for key, m in {"sim_30_50": (a > 30) & (a <= 50), "sim_90_100": a > 90, "sim_eq_100": a >= 99.99,
                       "sim_gt_70": a > 70, "sim_eq_0": a == 0}.items():
            if m.any():
                g[key] = {"n": int(m.sum()), "MAE": r[m].abs().mean(), "bias": r[m].mean()}
        m = d.active_facades == "east;south;west"
        g["east_south_west"] = {"n": int(m.sum()), "MAE": r[m].abs().mean(), "max_abs": r[m].abs().max()}
        out[t] = g
    north = data[data.active_facades == "north"]
    widths = np.linspace(0.6, 2.4, 37)
    ase = models["ASE_1000_250_percent"]
    out["north_only_ASE_max_pred_heldout"] = float(d.loc[d.active_facades == "north", "pred_ASE_1000_250_percent"].max())
    out["north_only_ASE_max_pred_sweep"] = float(max(common.clip(sweep(ase, r, feats, "window_width_m", widths)).max()
                                                     for _, r in north.iterrows()))
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data = common.load_dataset(common.DEFAULT_CASES, common.DEFAULT_RESULTS)
    train, test = common.split_dataset(data)
    feats = common.feature_columns("reduced")
    models = {t: joblib.load(ROOT / f"model_{t}.joblib") for t in common.TARGETS}
    d = heldout_predictions(data, train, test, feats, models)
    d.to_csv(ROOT / "heldout_predictions_cv_and_test.csv", index=False, float_format="%.6f")
    figure_residuals(d)
    figure_north_only(data, feats, models)
    nums = numbers(d, data, feats, models)
    (ROOT / "limitations_numbers.json").write_text(json.dumps(nums, indent=2, default=float), encoding="utf-8")
    print(json.dumps(nums, indent=1, default=lambda v: round(float(v), 2)))


if __name__ == "__main__":
    main()
