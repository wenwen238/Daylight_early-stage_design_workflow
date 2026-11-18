#!/usr/bin/env python3
"""Does the surrogate work for overhang depths between 0 and 0.30 m, where there is no training data?

One design (south window, conventional, 1.80 m wide x 1.50 m high) is simulated in Radiance at
overhang depths from 0 to 1.50 m, including six depths inside the 0-0.30 m gap, with the same
code and settings as the training data (2_radiance/annual_metrics.py --case-json). The MLP
ensemble is evaluated on a fine grid of depths for the same design. Comparing the two shows
how the prediction behaves where the model has no data.

Outputs (thesis_data/results/extensions/overhang_gap/): one Radiance folder per depth, a table and a figure.
Usage:  python 6_extensions/overhang_gap_check.py            (simulations + figure)
        python 6_extensions/overhang_gap_check.py --plot     (figure from the saved table)
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "3_ml", ROOT / "4_decision"):
    sys.path.insert(0, str(p))
import config  # noqa: E402
import common  # noqa: E402
import decision  # noqa: E402

OUT = config.RESULTS / "extensions" / "overhang_gap"
DESIGN = dict(active_facades=["south"], window_type="conventional", window_width_m=1.80, window_height_m=1.50)
DEPTHS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.45, 0.60, 0.90, 1.20, 1.50]
GAP = (0.0, 0.30)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ecebe7"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 9.5, "axes.titlecolor": INK, "axes.linewidth": 0.8,
})


def simulate(depth):
    name = f"overhang_{depth:.2f}".replace(".", "p")
    out = OUT / name
    if (out / "metrics.json").exists():
        m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    else:
        case = dict(decision.make_case(**DESIGN, overhang_depth_m=depth), model_id=name)
        case_file = OUT / f"{name}_case.json"
        case_file.write_text(json.dumps(case, indent=2), encoding="utf-8")
        proc = subprocess.run([sys.executable, str(ROOT / "2_radiance" / "annual_metrics.py"),
                               "--case-json", str(case_file), "--out", str(out)],
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, cwd=str(ROOT))
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-2000:])
        m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    return {"overhang_depth_m": depth, "radiance_sDA": m["sDA_300_50_percent"],
            "radiance_ASE": m["ASE_1000_250_percent"], "seconds": m.get("total_model_s")}


def ensemble(depths):
    """Ensemble mean and member range of the clipped predictions for every depth."""
    models = decision._models()
    X = pd.DataFrame([decision.make_case(**DESIGN, overhang_depth_m=d) for d in depths])
    for f, flag in zip(common.FACADES, common.FACADE_FLAGS):
        X[flag] = X["active_facades"].apply(lambda a, f=f: int(f in a))
    X = X[common.feature_columns("reduced")]
    out = {"overhang_depth_m": depths}
    for t, short in zip(common.TARGETS, ("sDA", "ASE")):
        p = np.array([common.clip(m.predict(X)) for m in models[t]])
        out[f"mlp_{short}"], out[f"mlp_{short}_min"], out[f"mlp_{short}_max"] = p.mean(0), p.min(0), p.max(0)
    return pd.DataFrame(out)


def figure(sim, curve):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for ax, short, label in zip(axes, ("sDA", "ASE"), (common.TARGET_LABELS[t] for t in common.TARGETS)):
        ax.axvspan(*GAP, color=BAND, linewidth=0, zorder=0)
        ax.fill_between(curve.overhang_depth_m, curve[f"mlp_{short}_min"], curve[f"mlp_{short}_max"],
                        color=BLUE, alpha=0.18, linewidth=0)
        ax.plot(curve.overhang_depth_m, curve[f"mlp_{short}"], color=BLUE, linewidth=1.8, label="MLP ensemble")
        ax.scatter(sim.overhang_depth_m, sim[f"radiance_{short}"], s=26, color=ORANGE, edgecolors="white",
                   linewidths=0.6, zorder=3, label="Radiance simulation")
        lo, hi = ax.get_ylim()
        ax.text(0.15, lo + 0.04 * (hi - lo), "no training\ndata", ha="center", va="bottom", fontsize=7,
                color=INK2)
        ax.set_title(label, loc="left")
        ax.set_xlabel("overhang depth [m]")
        ax.set_ylabel(f"{label} [%]")
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.18))
    labels.append("range of the five MLPs")
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "fig_overhang_gap_check.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--plot", action="store_true", help="only redraw the figure from the saved table")
    args = args.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.plot:
        sim = pd.read_csv(OUT / "overhang_gap_radiance.csv")
    else:
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            sim = pd.DataFrame(list(pool.map(simulate, DEPTHS)))
        pred = ensemble(DEPTHS)
        sim = sim.merge(pred, on="overhang_depth_m")
        sim.to_csv(OUT / "overhang_gap_radiance.csv", index=False, float_format="%.3f")
        print(sim[["overhang_depth_m", "radiance_sDA", "mlp_sDA", "radiance_ASE", "mlp_ASE"]].round(2).to_string(index=False))
        print(f"simulations: {time.perf_counter() - start:.0f}s")
    curve = ensemble(list(np.round(np.arange(0, 1.501, 0.01), 2)))
    figure(sim, curve)
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
