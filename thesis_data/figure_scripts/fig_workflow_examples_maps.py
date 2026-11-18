#!/usr/bin/env python3
"""Map figures of the two workflow examples from the independent test (Section 5.2), in plan view
(windows blue, overhangs orange, white criterion line).

  A046: accepted MLP prediction -> predicted 10 x 10 maps (PCA + MLP), from the saved tool result
  B003: Radiance simulation     -> simulated 32 x 32 maps, from the Radiance run of the tool

Reads thesis_data/results/extensions/workflow_examples/<id>_tool_result.json (written by workflow_examples_independent.py); no new run.

    python thesis_data/figure_scripts/fig_workflow_examples_maps.py

Output: thesis_data/results/extensions/workflow_examples/fig_<id>_da300_map.png and fig_<id>_ase_map.png
"""

import json
import sys
from pathlib import Path

import contourpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "3_ml"))
sys.path.insert(0, str(ROOT / "4_decision"))
import decision  # noqa: E402

DIR = ROOT / "thesis_data" / "results" / "extensions" / "workflow_examples"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
BLUE, ORANGE = "#2a78d6", "#eb6834"
W, D = 6.0, 6.0
MAPS = {
    "da300": dict(vmax=100, cmap="viridis", level=50, extend="neither", label="DA300 [% of occupied hours]",
                  outline=False, frame=INK),
    "ase": dict(vmax=1000, cmap="inferno", level=250, extend="max",
                label="Occupied hours with direct\nilluminance above 1000 lux [h]", outline=True, frame=MUTED),
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
TEXT_SIZE = 12


def load(name):
    r = json.loads((DIR / f"{name}_tool_result.json").read_text(encoding="utf-8"))
    i = r["inputs"] if "inputs" in r else None
    final = r["final"]
    if r.get("radiance_output"):                       # simulated 32 x 32 maps, rows from south
        run = Path(r["radiance_output"])
        if not run.is_absolute():                      # stored relative to the repository root
            run = ROOT / run
        maps = {"da300": np.load(run / "da300_map.npy") * 100, "ase": np.load(run / "ase_hours_map.npy")}
        case = json.loads(next(run.parent.glob(f"{run.name}_case.json")).read_text(encoding="utf-8"))
        source = "Radiance, 32 × 32"
    else:                                              # predicted 10 x 10 maps, rows from south
        m = r["mlp_maps"]
        maps = {"da300": np.array(m["da300"]["values"], float), "ase": np.array(m["ase_hours"]["values"], float)}
        case = None
        source = "predicted, PCA + MLP, 10 × 10"
    return r, final, maps, case, source, i


def plot(name, metric, values, case, final, source):
    spec = MAPS[metric]
    ny, nx = values.shape
    fig, ax = plt.subplots(figsize=(5.6, 4.9))
    im = ax.imshow(values, origin="lower", extent=[0, W, 0, D], vmin=0, vmax=spec["vmax"], cmap=spec["cmap"], zorder=1)
    xs, ys = (np.arange(nx) + 0.5) * W / nx, (np.arange(ny) + 0.5) * D / ny
    effects = [pe.Stroke(linewidth=2.4, foreground=INK), pe.Normal()] if spec["outline"] else None
    for line in contourpy.contour_generator(xs, ys, values).lines(spec["level"]):
        if np.linalg.norm(np.diff(line, axis=0), axis=1).sum() > 1.0:
            ax.plot(line[:, 0], line[:, 1], color="white", linewidth=1.2, zorder=2, path_effects=effects)
    ax.add_patch(Rectangle((0, 0), W, D, fill=False, edgecolor=spec["frame"], linewidth=1.2, zorder=3))
    for w in case["windows"]:
        u0, u1 = w["x0_m"], w["x1_m"]
        line = {"north": ([u0, u1], [D, D]), "south": ([u0, u1], [0, 0]),
                "west": ([0, 0], [u0, u1]), "east": ([W, W], [u0, u1])}[w["face"]]
        ax.plot(*line, color=BLUE, linewidth=4, solid_capstyle="butt", zorder=4)
    for s in case["shading"]:
        u0, u1, dep = s["x0_m"], s["x1_m"], s["depth_m"]
        if dep <= 0:
            continue
        xy, wh = {"north": ((u0, D), (u1 - u0, dep)), "south": ((u0, -dep), (u1 - u0, dep)),
                  "west": ((-dep, u0), (dep, u1 - u0)), "east": ((W, u0), (dep, u1 - u0))}[s["face"]]
        ax.add_patch(Rectangle(xy, *wh, facecolor=ORANGE, edgecolor="none", alpha=0.85, zorder=3))
    south = max([s["depth_m"] for s in case["shading"] if s["face"] == "south"], default=0.0)
    text_y = -0.45 if south == 0 else -south - 0.4
    ax.set_xlim(-1.5, W + 1.5)
    ax.set_ylim(text_y - 1.0, D + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.annotate("N", xy=(W + 0.9, D + 1.3), xytext=(W + 0.9, D + 0.65), ha="center", va="center",
                fontsize=TEXT_SIZE, color=INK, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.0))
    ax.text(0.02, text_y + 0.15, f"sDA300/50% = {final['sDA_300_50_percent']:.1f}%\n"
                                f"ASE1000,250h = {final['ASE_1000_250_percent']:.1f}%",
            ha="left", va="top", fontsize=TEXT_SIZE, color=INK, linespacing=1.4)
    ax.set_title(f"{name}: map {source}", fontsize=TEXT_SIZE, color=INK2, loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend=spec["extend"])
    cbar.set_label(spec["label"], color=INK2)
    cbar.ax.axhline(spec["level"], color="white", linewidth=1.2)
    cbar.outline.set_edgecolor(MUTED)
    out = DIR / f"fig_{name}_{metric}_map.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out)


def main():
    specs = {"A046": dict(active_facades=["north", "west"], window_type="floor_to_ceiling", window_width_m=1.05,
                          overhang_depth_m=0.62),
             "B003": dict(active_facades=["north", "south"], window_type="floor_to_ceiling", window_width_m=0.70,
                          overhang_depth_m=1.28)}
    for name, design in specs.items():
        r, final, maps, case, source, _ = load(name)
        case = case or decision.make_case(design["active_facades"], design["window_type"], design["window_width_m"],
                                          None, None, design["overhang_depth_m"])
        for metric in ("da300", "ase"):
            plot(name, metric, maps[metric], case, final, source)


if __name__ == "__main__":
    main()
