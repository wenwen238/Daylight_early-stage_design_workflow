#!/usr/bin/env python3
"""Thesis figures: randomised hyperparameter search of the MLP and of XGBoost (60 configurations per target).

The saved cv_results files are ranked, not in search order. RandomizedSearchCV draws
its configurations with ParameterSampler and the fixed random_state, so the search
order is rebuilt here by drawing the same 60 configurations again and matching them
to the saved CV scores (nothing is retrained).

The MLP figure shows the best XGBoost CV RMSE as a reference line; the XGBoost
figure stands on its own.
"""

import ast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "3_ml"))
import common  # noqa: E402
import train_mlp
import train_xgboost

FEATURE_SET = "reduced"
MLP_COLOR, XGB_COLOR = "#1baf7a", "#eb6834"          # as in fig_model_comparison_test.png
INK, INK2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ecebe7"
# key: (label, colour, search space, figure name, show XGBoost reference line)
MODELS = {
    "mlp": ("MLP", MLP_COLOR, train_mlp.SEARCH_SPACE, "fig_mlp_hyperparameter_search.png", True),
    "xgboost": ("XGBoost", XGB_COLOR, train_xgboost.SEARCH_SPACE, "fig_xgboost_hyperparameter_search.png", False),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 9.5, "axes.titlecolor": INK, "axes.linewidth": 0.8,
})


def normalise(value):
    """Make a sampled value and its CSV text compare equal (CSV floats have 6 decimals)."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value)
    return round(float(value), 6)


def search_order(root, key, space, target, n_iter):
    """CV results of one target in the order the configurations were drawn."""
    cv = pd.read_csv(root / key / f"cv_results_{target}.csv")
    names = [p.split("__")[-1] for p in space]
    cv["key"] = [tuple(normalise(v) for v in row) for row in cv[names].itertuples(index=False)]
    drawn = ParameterSampler(space, n_iter, random_state=common.RANDOM_STATE)
    keys = [tuple(normalise(s[p]) for p in space) for s in drawn]
    ordered = cv.set_index("key").loc[keys].reset_index(drop=True)
    if len(ordered) != len(cv):
        raise ValueError(f"{key} {target}: rebuilt search order does not match the saved CV results")
    ordered.insert(0, "trial", np.arange(1, len(ordered) + 1))
    return ordered


def plot_model(root, out, key, show_sd=True):
    label, color, space, filename, show_xgb = MODELS[key]
    if not show_sd:
        filename = filename.replace(".png", "_no_sd.png")
    xgb = pd.read_csv(root / "xgboost" / "cv_summary.csv").set_index("target")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for ax, target in zip(axes, common.TARGETS):
        d = search_order(root, key, space, target, n_iter=60)
        rmse = d["mean_test_rmse"].to_numpy()
        best_i = int(rmse.argmin())
        best, best_sd = rmse[best_i], d["std_test_rmse"].iloc[best_i]
        n_close = int((rmse <= best + best_sd).sum())

        if show_sd:
            ax.axhspan(best - best_sd, best + best_sd, color=BAND, zorder=0, linewidth=0)
        ax.scatter(d["trial"], rmse, s=16, color=color, alpha=0.55, linewidths=0.4,
                   edgecolors="white", zorder=2)
        ax.step(d["trial"], np.minimum.accumulate(rmse), where="post", color=INK, linewidth=1.4, zorder=3)
        ax.scatter([best_i + 1], [best], s=46, color=color, edgecolors=INK, linewidths=1.0, zorder=4)
        hi = rmse.max()
        if show_xgb:
            xgb_rmse = xgb.loc[target, "cv_RMSE_mean"]
            ax.axhline(xgb_rmse, color=XGB_COLOR, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
            hi = max(hi, xgb_rmse)
        note = (f"selected: configuration {best_i + 1}\nCV RMSE {best:.2f} ± {best_sd:.2f} pp\n"
                f"{n_close} configurations within ± 1 SD" if show_sd
                else f"selected: configuration {best_i + 1}\nCV RMSE {best:.2f} pp")
        ax.text(0.98, 0.97, note, transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=INK2, linespacing=1.4)

        ax.set_ylim(0, hi * 1.42)
        ax.set_xlim(0, 61)
        ax.set_title(common.TARGET_LABELS[target], fontsize=9.5, weight="bold", color=INK)
        ax.set_xlabel("search configuration (in order drawn)")
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        print(f"{label} {target}: best trial {best_i + 1}, CV RMSE {best:.3f} ± {best_sd:.3f}, "
              f"{n_close} within 1 SD, median {np.median(rmse):.3f}, max {rmse.max():.3f}")
    axes[0].set_ylabel("5-fold CV RMSE [pp]")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=color, markersize=5, label=f"{label} configuration"),
               plt.Line2D([], [], color=INK, linewidth=1.4, label="best so far")]
    if show_sd:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=BAND, label="selected ± 1 fold SD"))
    if show_xgb:
        handles.append(plt.Line2D([], [], color=XGB_COLOR, linewidth=1.2, linestyle=(0, (4, 3)),
                                  label="best XGBoost"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(w_pad=1.5)
    path = out / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(path)


def main():
    root = common.OUT_ROOT / FEATURE_SET
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for key in MODELS:
        plot_model(root, out, key)


if __name__ == "__main__":
    main()
