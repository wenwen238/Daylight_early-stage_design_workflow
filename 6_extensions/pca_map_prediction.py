#!/usr/bin/env python3
"""Predict the whole map, not only sDA / ASE (10 x 10 grid, PCA + MLP; thesis Section 4.3.4).

Two maps are available for every one of the 560 Radiance cases:
  --metric da300 (default)  DA300 map: share of occupied hours with at least 300 lx  [%]
                            -> sDA300/50% = share of points with DA300 >= 50 %
  --metric ase              direct-sun hours above 1000 lx (occupied hours)          [h]
                            -> ASE1000,250h = share of points with more than 250 h

Grid
  The stored Radiance maps have 32 x 32 sensors at the cell centres of the 6 x 6 m work plane
  (spacing 0.1875 m). They are resampled by bilinear interpolation to the 10 x 10 cell centres of
  a 0.6 m grid (x, y = 0.3, 0.9, ..., 5.7 m), which cover the whole floor. All points lie inside the
  32 x 32 grid, so no extrapolation is needed and no new Radiance run is required. sDA and ASE of
  the thesis stay those of the 32 x 32 grid.

Method
  1. PCA of the 10 x 10 maps of the 448 training cases (same split as 3_ml/common.py).
     Each map = mean map + sum of k component patterns x scores.
  2. An MLP predicts the k scores from the eight design inputs; five seeds are averaged, as for
     the sDA/ASE surrogate. The network uses the hyperparameters selected for the matching
     scalar model (sDA for the DA300 map, ASE for the ASE map; reduced/mlp/best_params.json);
     they are not re-tuned for the maps.
  3. The predicted map is the PCA reconstruction of the predicted scores, limited to the valid range.
  k is chosen from the training data alone, by five-fold cross-validation (the k with the lowest
  CV RMSE); the k that explains 99.5 % of the variance is reported for comparison.
  For comparison, other k values and an MLP that predicts the map points directly are also run.

Outputs (thesis_data/results/extensions/pca_map_da300_10x10/ for DA300,
thesis_data/results/extensions/pca_map_ase_10x10/ for ASE): tables, the maps, and four figures.

    python 6_extensions/pca_map_prediction.py --metric da300
    python 6_extensions/pca_map_prediction.py --metric ase
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "3_ml"))
import config  # noqa: E402
import common  # noqa: E402
import train_mlp  # noqa: E402

METRICS = {
    "da300": dict(out="pca_map_da300", file="da300_map.npy", scale=100.0, valid=(0.0, 100.0), unit="pp",
                  title="DA300 map: share of working hours with at least 300 lx at each point",
                  value_label="DA300 [%]", cmap="viridis", vmax=100.0, target="sDA_300_50_percent",
                  threshold=50.0, at_least=True, maps_csv="da300_maps_10x10.csv", var_ylim=(90, 100.3)),
    "ase": dict(out="pca_map_ase", file="ase_hours_map.npy", scale=1.0, valid=(0.0, 3650.0), unit="h",
                title="ASE map: working hours with direct sun above 1000 lx at each point",
                value_label="direct sun > 1000 lx [h]", cmap="inferno", vmax=1500.0,
                target="ASE_1000_250_percent", threshold=250.0, at_least=False,
                maps_csv="ase_hours_maps_10x10.csv", var_ylim=(70, 100.3)),
}
CFG = METRICS["da300"]
OUT = config.RESULTS / "extensions" / (CFG["out"] + "_10x10")

W = D = 6.0
N32 = 32
X32 = (np.arange(N32) + 0.5) * W / N32               # sensor positions of the Radiance grid
XG = (np.arange(10) + 0.5) * 0.6                     # map points [m]: cell centres 0.3 ... 5.7
EXTENT = (0.0, W)                                    # floor area the map shows [m]
NG = len(XG)
SEEDS = [20260924, 20260925, 20260926, 20260927, 20260928]
VAR_TARGET = 0.995
NL_ = "\n"                                           # line break in legend labels
K_LIST = [1, 2, 3, 5, 8, 10, 15, 20]

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 9, "axes.titlecolor": INK, "axes.linewidth": 0.8,
})


# -- data ---------------------------------------------------------------------

def to_grid(map32):
    """Bilinear interpolation of a 32 x 32 map (rows = y from south, columns = x from west)."""
    f = RegularGridInterpolator((X32, X32), map32, method="linear")
    yy, xx = np.meshgrid(XG, XG, indexing="ij")
    return f(np.stack([yy.ravel(), xx.ravel()], axis=1)).reshape(NG, NG)


def load_maps(data):
    maps = np.array([to_grid(np.load(config.RADIANCE_560 / m / CFG["file"]).astype(float))
                     for m in data.model_id])
    return maps.reshape(len(data), -1) * CFG["scale"]


def meets(maps):
    """Points that count towards the metric: DA300 >= 50 % (sDA) or more than 250 h (ASE)."""
    return maps >= CFG["threshold"] if CFG["at_least"] else maps > CFG["threshold"]


def metric_from_map(maps):
    """sDA (share of points with DA300 >= 50 %) or ASE (share with more than 250 h) of the coarse grid, in %."""
    return meets(maps).mean(axis=1) * 100


def clip(m):
    return np.clip(m, *CFG["valid"])


# -- models -------------------------------------------------------------------

# cfg is passed to the joblib workers explicitly: a worker re-imports this module with the default
# CFG (DA300), so a metric chosen by another script through pmp.CFG would not reach it.

def best_params(cfg=None):
    p = json.loads((config.ML_TRAINING / "reduced" / "mlp" / "best_params.json").read_text(encoding="utf-8"))
    return {k: tuple(v) if isinstance(v, list) else v for k, v in p[(cfg or CFG)["target"]].items()}


def fit_predict(Xtr, Ytr, Xte, seed, cfg=None):
    m = train_mlp.make_estimator("reduced").set_params(**best_params(cfg),
                                                       regressor__regressor__random_state=seed)
    m.fit(Xtr, Ytr)
    p = m.predict(Xte)
    return p.reshape(len(Xte), -1)


def ensemble_predict(Xtr, Ytr, Xte):
    preds = Parallel(n_jobs=5)(delayed(fit_predict)(Xtr, Ytr, Xte, s, CFG) for s in SEEDS)
    return np.mean(preds, axis=0)


def cv_one(k, fold, X, Y, cfg=None):
    """Point RMSE of PCA(k) + MLP (first seed) on one cross-validation fold of the training cases."""
    tr, va = fold
    pca = PCA(n_components=k).fit(Y[tr])
    s = fit_predict(X.iloc[tr], pca.transform(Y[tr]), X.iloc[va], SEEDS[0], cfg)
    pred = np.clip(pca.inverse_transform(s), *(cfg or CFG)["valid"])
    return float(np.sqrt(((pred - Y[va]) ** 2).mean()))


def map_scores(true, pred, metric32):
    u = CFG["unit"]
    err = pred - true
    per_map = np.sqrt((err ** 2).mean(axis=1))
    m_true, m_pred = metric_from_map(true), metric_from_map(pred)
    base = "sDA" if CFG["at_least"] else "ASE"
    return {f"point_RMSE_{u}": float(np.sqrt((err ** 2).mean())), f"point_MAE_{u}": float(np.abs(err).mean()),
            f"map_RMSE_median_{u}": float(np.median(per_map)), f"map_RMSE_p90_{u}": float(np.quantile(per_map, 0.9)),
            f"map_RMSE_max_{u}": float(per_map.max()),
            "points_correct_side_percent": float((meets(pred) == meets(true)).mean() * 100),
            f"{base}{NG}_MAE_pp": float(np.abs(m_pred - m_true).mean()),
            f"{base}{NG}_vs_{base}32_MAE_pp": float(np.abs(m_true - metric32).mean())}


# -- figures ------------------------------------------------------------------

def draw_windows(ax, case):
    """Windows as thick lines just outside the plan, in map coordinates."""
    for w in case["windows"]:
        a, b = w["x0_m"], w["x1_m"]
        seg = {"south": ([a, b], [-0.25, -0.25]), "north": ([W - b, W - a], [D + 0.25, D + 0.25]),
               "east": ([W + 0.25, W + 0.25], [a, b]), "west": ([-0.25, -0.25], [D - b, D - a])}[w["face"]]
        ax.plot(*seg, color=BLUE, linewidth=2.6, solid_capstyle="butt", clip_on=False)


def show(ax, m, vmin, vmax, cmap):
    return ax.imshow(m.reshape(NG, NG), origin="lower", extent=EXTENT * 2, vmin=vmin, vmax=vmax,
                     cmap=cmap, interpolation="nearest")


def plan_axes(ax):
    ax.set_xlim(-0.45, W + 0.45)
    ax.set_ylim(-0.45, D + 0.45)
    ax.plot([0, W, W, 0, 0], [0, 0, D, D, 0], color=INK2, linewidth=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fig_components(pca, mean):
    fig = plt.figure(figsize=(7.4, 2.5))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 0.07], hspace=0.35, wspace=0.15)
    axes = np.array([fig.add_subplot(gs[0, c]) for c in range(5)])
    cax_mean, cax_pc = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1:])
    im = show(axes[0], mean, 0, CFG["vmax"], CFG["cmap"])
    axes[0].set_title("average map", fontsize=8.5)
    plan_axes(axes[0])
    lim = np.abs(pca.components_[:4]).max()
    for i, ax in enumerate(axes[1:]):
        im2 = show(ax, pca.components_[i], -lim, lim, "RdBu_r")
        ax.set_title(f"pattern {i + 1}: {pca.explained_variance_ratio_[i] * 100:.0f} %", fontsize=8.5)
        plan_axes(ax)
    for ax in axes:
        ax.text(W / 2, -0.42, "S", ha="center", va="top", fontsize=6.5, color=MUTED)
    cb = fig.colorbar(im, cax=cax_mean, orientation="horizontal")
    cb.ax.tick_params(labelsize=6.5)
    cb.set_label(CFG["value_label"], fontsize=7)
    cb2 = fig.colorbar(im2, cax=cax_pc, orientation="horizontal")
    cb2.ax.tick_params(labelsize=6.5)
    cb2.set_label("pattern value: red adds to the average map, blue subtracts (for a positive weight)",
                  fontsize=7)
    fig.suptitle(CFG["title"].split(":")[0] + ": average map and the four largest patterns "
                 "(% = share of the variation between maps)", fontsize=8.5, y=1.0)
    fig.savefig(OUT / "fig_pca_components.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_components_all(pca, mean, ncols=6):
    """Average map and all k selected patterns, six panels per row."""
    k = len(pca.components_)
    rows = int(np.ceil((k + 1) / ncols))
    fig = plt.figure(figsize=(7.4, 1.6 * rows + 0.55))
    gs = fig.add_gridspec(rows + 1, ncols, height_ratios=[1] * rows + [0.07], hspace=0.45, wspace=0.15)
    axes = [fig.add_subplot(gs[i // ncols, i % ncols]) for i in range(k + 1)]
    cax_mean, cax_pc = fig.add_subplot(gs[rows, 0]), fig.add_subplot(gs[rows, 1:])
    im = show(axes[0], mean, 0, CFG["vmax"], CFG["cmap"])
    axes[0].set_title("average\nmap", fontsize=8)
    lim = np.abs(pca.components_).max()
    for i, ax in enumerate(axes[1:]):
        im2 = show(ax, pca.components_[i], -lim, lim, "RdBu_r")
        share = pca.explained_variance_ratio_[i] * 100
        digits = 0 if share >= 0.95 else 1 if share >= 0.095 else 2     # as printed after rounding
        ax.set_title(f"pattern {i + 1}\n{share:.{digits}f} %", fontsize=8)
    for ax in axes:
        plan_axes(ax)
        ax.text(W / 2, -0.42, "S", ha="center", va="top", fontsize=6, color=MUTED)
    cb = fig.colorbar(im, cax=cax_mean, orientation="horizontal")
    cb.ax.tick_params(labelsize=6.5)
    cb.set_label(CFG["value_label"], fontsize=7)
    cb2 = fig.colorbar(im2, cax=cax_pc, orientation="horizontal")
    cb2.ax.tick_params(labelsize=6.5)
    cb2.set_label("pattern value: red adds to the average map, blue subtracts (for a positive weight)",
                  fontsize=7)
    fig.suptitle(CFG["title"].split(":")[0] + f": average map and all {k} patterns of the model "
                 f"(together {pca.explained_variance_ratio_.sum() * 100:.1f} % of the variation between maps)",
                 fontsize=8.5, y=0.995)
    fig.savefig(OUT / "fig_pca_components_all.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_k(var_table, k_table, k_sel):
    u = CFG["unit"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    ax = axes[0]
    ax.plot(var_table.k, var_table.cumulative_variance * 100, color=INK, linewidth=1.6, marker="o", markersize=3)
    ax.axhline(VAR_TARGET * 100, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.axvline(k_sel, color=ORANGE, linewidth=1.0)
    ax.set_ylim(*CFG["var_ylim"])
    ax.text(k_sel + 0.4, CFG["var_ylim"][0] + 1, f"k = {k_sel}", color=ORANGE, fontsize=7.5)
    ax.set_xlabel("number of patterns k")
    ax.set_ylabel("variation of the training maps\nexplained by k patterns [%]")
    ax = axes[1]
    ax.plot(k_table.k, k_table[f"reconstruction_RMSE_{u}"], color=MUTED, linewidth=1.4, marker="o", markersize=3,
            label="k patterns with" + NL_ + "their true weights")
    ax.plot(k_table.k, k_table[f"cv_RMSE_{u}"], color=GREEN, linewidth=1.2, linestyle=(0, (2, 2)), marker="s",
            markersize=3, label="PCA + MLP," + NL_ + "cross-validation")
    ax.plot(k_table.k, k_table[f"predicted_RMSE_{u}"], color=GREEN, linewidth=1.8, marker="o", markersize=3.5,
            label="PCA + MLP," + NL_ + "test cases")
    ax.axhline(k_table[f"direct_RMSE_{u}"].iloc[0], color=BLUE, linewidth=1.2, linestyle=(0, (4, 3)),
               label=f"MLP, {NG * NG} points" + NL_ + "directly, test cases")
    ax.axvline(k_sel, color=ORANGE, linewidth=1.0)
    ax.set_xlabel("number of patterns k")
    ax.set_ylabel(f"error per point (RMSE) [{u}]")
    handles, labels = ax.get_legend_handles_labels()
    for a in axes:
        a.grid(True, color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout(w_pad=2.0)
    axes[1].legend(handles, labels, frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5),
                   handlelength=2.4, labelspacing=1.1)
    fig.savefig(OUT / "fig_pca_k.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_examples(test, true, pred, cases, pred_title="Predicted\n(PCA + MLP)", out=None,
                 filename="fig_pca_map_examples.png"):
    u = CFG["unit"]
    per_map = np.sqrt(((pred - true) ** 2).mean(axis=1))
    order = np.argsort(per_map)
    picks = [order[len(order) // 10], order[len(order) // 2], order[-1]]
    labels = ["Small error (better than 90 % of the test cases)", "Typical error (median test case)",
              "Largest error in the test set"]
    lim = np.ceil(np.abs(pred[picks] - true[picks]).max())
    # three equal map rows and one colour-bar row, so all nine panels have the same size and position
    fig = plt.figure(figsize=(5.6, 1.95 * len(picks) + 0.9))
    outer = fig.add_gridspec(2, 1, height_ratios=[len(picks), 0.08], hspace=0.13)
    gs = outer[0].subgridspec(len(picks), 3, hspace=0.75, wspace=0.12)
    gc = outer[1].subgridspec(1, 3, wspace=0.12)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(len(picks))])
    cax_map, cax_diff = fig.add_subplot(gc[0, :2]), fig.add_subplot(gc[0, 2])
    for r, (i, lab) in enumerate(zip(picks, labels)):
        mid = test.model_id.iloc[i]
        case = cases[mid]
        im = show(axes[r, 0], true[i], 0, CFG["vmax"], CFG["cmap"])
        show(axes[r, 1], pred[i], 0, CFG["vmax"], CFG["cmap"])
        imd = show(axes[r, 2], pred[i] - true[i], -lim, lim, "RdBu_r")
        for c in range(3):
            plan_axes(axes[r, c])
            draw_windows(axes[r, c], case)
        kind = "floor-to-ceiling" if case["window_type"] == "floor_to_ceiling" else "conventional"
        shade = f"{case['overhang_depth_m']:.2f} m overhang" if case["overhang_depth_m"] > 0 else "no overhang"
        desc = (f"{lab} – {mid}  (error of this map: RMSE {per_map[i]:.1f} {u})\n"
                f"windows: {' + '.join(case['active_facades'])}; {kind}, "
                f"{case['window_width_m']:.2f} m wide; {shade}")
        axes[r, 0].text(-0.45, D + 0.75, desc, ha="left", va="bottom", fontsize=7, color=INK)
    # panel labels under every map, below the south-window bar, so each row reads on its own
    for r in range(len(picks)):
        for c, t in enumerate(["Simulated (Radiance)", pred_title.replace("\n", " "),
                               "Error (predicted − simulated)"]):
            axes[r, c].text(W / 2, -0.62, t, ha="center", va="top", fontsize=7, color=INK2)
    fig.suptitle(CFG["title"], fontsize=9, y=0.975)
    cb = fig.colorbar(im, cax=cax_map, orientation="horizontal",
                      extend="max" if CFG["vmax"] < CFG["valid"][1] else "neither")
    cb.set_label(CFG["value_label"], fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb2 = fig.colorbar(imd, cax=cax_diff, orientation="horizontal")
    cb2.set_label(f"error [{u}]  (red: too high, blue: too low)", fontsize=7)
    cb2.ax.tick_params(labelsize=6.5)
    fig.savefig((out or OUT) / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -- main ---------------------------------------------------------------------

def main():
    global CFG, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=list(METRICS), default="da300")
    args = parser.parse_args()
    CFG = METRICS[args.metric]
    tag = f"{NG}x{NG}"
    OUT = config.RESULTS / "extensions" / (CFG["out"] + f"_{tag}")
    u = CFG["unit"]

    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    data = common.load_dataset(common.DEFAULT_CASES, common.DEFAULT_RESULTS)
    feats = common.feature_columns("reduced")
    maps = pd.DataFrame(load_maps(data), index=data.model_id)
    train, test = common.split_dataset(data)
    Ytr, Yte = maps.loc[train.model_id].to_numpy(), maps.loc[test.model_id].to_numpy()
    Xtr, Xte = train[feats], test[feats]
    metric32_te = test[CFG["target"]].to_numpy()
    maps.to_csv(OUT / CFG["maps_csv"], float_format="%.3f")

    full = PCA().fit(Ytr)
    cum = np.cumsum(full.explained_variance_ratio_)
    k_var = int(np.argmax(cum >= VAR_TARGET) + 1)
    var_table = pd.DataFrame({"k": np.arange(1, 31), "cumulative_variance": cum[:30]})

    # k by five-fold cross-validation on the training cases (same folds as the thesis models)
    folds = common.cv_folds(train)
    k_all = sorted(set(K_LIST + [k_var]))
    jobs = [(k, f) for k in k_all for f in range(len(folds))]
    cv_rmse = Parallel(n_jobs=common.N_JOBS)(delayed(cv_one)(k, folds[f], Xtr, Ytr, CFG) for k, f in jobs)
    cv = pd.DataFrame([{"k": k, "fold": f, "rmse": r} for (k, f), r in zip(jobs, cv_rmse)])
    cv = cv.groupby("k").rmse.agg(["mean", "std"]).reset_index()
    cv.columns = ["k", f"cv_RMSE_{u}", f"cv_RMSE_sd_{u}"]
    cv.to_csv(OUT / "pca_k_cv.csv", index=False, float_format="%.4f")
    k_sel = int(cv.loc[cv[f"cv_RMSE_{u}"].idxmin(), "k"])
    print(cv.round(2).to_string(index=False), f"\nk selected by CV: {k_sel} (99.5 % variance would give {k_var})")
    var_table.to_csv(OUT / "pca_explained_variance.csv", index=False, float_format="%.6f")

    direct = clip(ensemble_predict(Xtr, Ytr, Xte))
    direct_scores = map_scores(Yte, direct, metric32_te)

    rows, preds = [], {}
    for k in k_all:
        pca = PCA(n_components=k).fit(Ytr)
        recon = clip(pca.inverse_transform(pca.transform(Yte)))
        pred = clip(pca.inverse_transform(ensemble_predict(Xtr, pca.transform(Ytr), Xte)))
        preds[k] = (pca, pred)
        s = map_scores(Yte, pred, metric32_te)
        rows.append({"k": k, "cumulative_variance": cum[k - 1],
                     f"reconstruction_RMSE_{u}": map_scores(Yte, recon, metric32_te)[f"point_RMSE_{u}"],
                     f"predicted_RMSE_{u}": s[f"point_RMSE_{u}"], f"direct_RMSE_{u}": direct_scores[f"point_RMSE_{u}"],
                     **{f"pred_{a}": b for a, b in s.items() if a != f"point_RMSE_{u}"}})
        print(f"k={k:2d}  var {cum[k - 1] * 100:6.2f} %  recon {rows[-1][f'reconstruction_RMSE_{u}']:.2f}  "
              f"pred {s[f'point_RMSE_{u}']:.2f} {u}", flush=True)
    k_table = pd.DataFrame(rows)
    k_table.to_csv(OUT / "pca_k_comparison.csv", index=False, float_format="%.4f")

    pca, pred = preds[k_sel]
    sel = map_scores(Yte, pred, metric32_te)
    summary = {"metric": CFG["file"].replace(".npy", ""), "unit": u,
               "grid": f"{NG} x {NG} points, 0.6 m spacing ({XG[0]:.1f}-{XG[-1]:.1f} m), "
                       "bilinear from the 32 x 32 Radiance maps",
               "k_selected_by_cv": k_sel, "k_for_99_5_percent_variance": k_var,
               "explained_variance_k": float(cum[k_sel - 1]),
               "pca_mlp": sel, f"direct_mlp_{NG * NG}_outputs": direct_scores,
               "test_map_mean": float(Yte.mean()), "test_share_of_zero_points": float((Yte == 0).mean()),
               "n_train": len(train), "n_test": len(test), "seeds": SEEDS,
               "elapsed_s": time.perf_counter() - start}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(pred, index=test.model_id).to_csv(OUT / f"test_predicted_maps_{tag}.csv", float_format="%.3f")
    pd.DataFrame(direct, index=test.model_id).to_csv(OUT / f"test_predicted_maps_{tag}_direct.csv",
                                                     float_format="%.3f")

    # components with a sign that makes the mean loading positive (PCA signs are arbitrary)
    for i in range(len(pca.components_)):
        if pca.components_[i].sum() < 0:
            pca.components_[i] *= -1
    cases = {c["model_id"]: c for c in json.loads(config.CASES_JSON.read_text(encoding="utf-8"))["cases"]}
    fig_components(pca, pca.mean_)
    fig_components_all(pca, pca.mean_)
    fig_k(var_table, k_table.merge(cv, on="k"), k_sel)
    fig_examples(test.reset_index(drop=True), Yte, pred, cases)
    fig_examples(test.reset_index(drop=True), Yte, direct, cases, pred_title="Predicted\n(point-wise MLP)",
                 filename="fig_pointwise_map_examples.png")
    print(json.dumps(summary, indent=1))
    print(f"Saved to {OUT}  ({time.perf_counter() - start:.0f}s)")


if __name__ == "__main__":
    main()
