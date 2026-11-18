"""MCP server: daylight decision workflow + Rhino drawing.

Started automatically by Claude Code through .mcp.json (do not run it by hand).

    Claude  --MCP-->  this server  --HTTP-->  rhino_bridge.py (inside Rhino)  -->  Rhino 8
                           |
                           +-- 4_decision/decision.py   (coverage check, MLP ensemble, threshold rule)
                           +-- 2_radiance/annual_metrics.py --case-json   (Radiance, separate process)
                           +-- 5_mcp/report.py   (fixed report, shown unchanged by the assistant)

Workflow tools: evaluate_design, suggest_changes, run_radiance, draw_design, get_sample_case,
design_space_info.
General Rhino tools (create_box, create_layer, call_rhino_tool, ...) follow below them.
Nothing in this file may print to stdout: stdout carries the MCP messages.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import requests
from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "3_ml", ROOT / "4_decision", ROOT / "2_radiance"):
    sys.path.insert(0, str(p))
import config  # noqa: E402
import decision  # noqa: E402
import map_prediction  # noqa: E402
import radiance_scene  # noqa: E402
import report  # noqa: E402

RHINO_URL = config.RHINO_URL
RADIANCE_TIMEOUT_S = 900

mcp = FastMCP("Rhino-MCP-Agent", instructions=(
    'When a tool result contains "report", show this report to the user in full and unchanged. '
    "Do not change, recalculate or round any value, do not reorder its parts, and do not translate it: "
    "the report stays in English whatever language the user writes in. "
    'Any comment of your own goes after the report, under the heading "Assistant comment (not calculated)".'))


def call_rhino(tool: str, args: dict):
    response = requests.post(
        RHINO_URL,
        json={"tool": tool, "args": args},
        timeout=20
    )
    response.raise_for_status()
    return response.json()


# -- workflow helpers -----------------------------------------------------------

def _case(active_facades, window_type, window_width_m, window_height_m, sill_height_m, overhang_depth_m,
          model_id="design"):
    return decision.make_case(active_facades, window_type, window_width_m, window_height_m,
                              sill_height_m, overhang_depth_m, model_id=model_id)


def _run_radiance(case):
    """Simulate one design with the same code and settings as the 560 training cases."""
    name = time.strftime("design_%Y%m%d_%H%M%S")
    out = config.MCP_RUNS / name
    out.parent.mkdir(parents=True, exist_ok=True)
    case = dict(case, model_id=name)
    case_file = config.MCP_RUNS / f"{name}_case.json"
    case_file.write_text(json.dumps(case, indent=2), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "2_radiance" / "annual_metrics.py"), "--case-json", str(case_file),
         "--out", str(out)],
        # stdin=DEVNULL: an inherited MCP stdin pipe makes the child hang on Windows
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=RADIANCE_TIMEOUT_S, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError("Radiance run failed:\n" + proc.stderr[-3000:])
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    metrics["out_dir"] = str(out)
    metrics["da300_map"] = np.load(out / "da300_map.npy")
    metrics["ase_hours_map"] = np.load(out / "ase_hours_map.npy")
    return metrics


LAYERS = {   # material in room.rad -> (layer, colour)
    "wall_mat": ("Daylight::Walls", (226, 226, 222)),
    "ceiling_mat": ("Daylight::Ceiling", (236, 236, 232)),
    "floor_mat": ("Daylight::Floor", (178, 178, 174)),
    "glass": ("Daylight::Glazing", (120, 185, 235)),
    "overhang_mat": ("Daylight::Overhang", (235, 125, 35)),
}
HEATMAP_LAYER = ("Daylight::DA300 map", (255, 255, 255))
CONTOUR_LAYER = ("Daylight::DA300 50% line", (255, 255, 255))
ASE_HEATMAP_LAYER = ("Daylight::ASE map", (255, 255, 255))
ASE_CONTOUR_LAYER = ("Daylight::ASE 250 h line", (0, 220, 255))
ASE_CMAP = matplotlib.colormaps["inferno"]             # same colours as the thesis ASE figures
ASE_VMAX = 1500.0                                       # hours; darker = fewer hours of direct sun
ASE_HOURS = 250
ASE_OFFSET_M = 8.0                                      # the ASE room is drawn 2 m east of the 6 m room
LABEL_LAYER = ("Daylight::Result", (0, 0, 0))
FACES = ("south", "north", "east", "west")
CMAP = matplotlib.colormaps["viridis"]                 # same colours as the thesis DA300 figure


def _rgb(fractions):
    return [[int(round(c * 255)) for c in CMAP(float(f))[:3]] for f in fractions]


def _rectangle(points):
    """Bounding rectangle of points lying in one vertical façade plane."""
    p = np.array(points)
    lo, hi = p.min(0), p.max(0)
    if hi[1] - lo[1] < 1e-9:                           # south / north façade (constant y)
        return [[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], lo[1], hi[2]], [lo[0], lo[1], hi[2]]]
    return [[lo[0], lo[1], lo[2]], [lo[0], hi[1], lo[2]], [lo[0], hi[1], hi[2]], [lo[0], lo[1], hi[2]]]


def _draw_walls(polygons, rgb, holes_supported=True, dx=0.0):
    """One surface per façade with the window as a hole; fall back to the room.rad pieces
    when the window touches the façade edge (floor-to-ceiling windows) or the bridge is an
    older version without holes."""
    layer = LAYERS["wall_mat"][0]
    n = 0
    for face in FACES:
        parts = [p for p in polygons if p["name"].split("_")[0] == face]
        walls = [p for p in parts if p["material"] == "wall_mat"]
        glass = [p for p in parts if p["material"] == "glass"]
        outline = _rectangle([q for p in parts for q in p["points"]])
        holes = [_rectangle(g["points"]) for g in glass]
        o = np.array(outline)
        planar_axes = [a for a in range(3) if np.ptp(o[:, a]) > 1e-9]       # the two in-plane axes
        touches = any(np.isclose(np.array(h)[:, a], o[:, a].min()).any() or
                      np.isclose(np.array(h)[:, a], o[:, a].max()).any() for h in holes for a in planar_axes)
        if glass and not touches and holes_supported:
            call_rhino("create_planar_surface", {"points": _shift(outline, dx),
                                                 "holes": [_shift(h, dx) for h in holes], "layer": layer,
                                                 "r": rgb[0], "g": rgb[1], "b": rgb[2]})
            n += 1
        else:
            for p in walls:
                call_rhino("create_planar_surface", {"points": _shift(p["points"], dx), "layer": layer,
                                                     "r": rgb[0], "g": rgb[1], "b": rgb[2]})
                n += 1
    return n


def _shift(points, dx):
    return [[p[0] + dx, p[1], p[2]] for p in points]


def _grid(n, length):
    """Origin, cell size and point positions of an n x n map at the cell centres: the 32 x 32
    Radiance grid or the 10 x 10 grid of the predicted maps (0.3 ... 5.7 m)."""
    return 0.0, length / n, (np.arange(n) + 0.5) * length / n


def _draw_map(values, kind, W, D, dx=0.0):
    """Map on the work plane and its criterion line.
    kind "da300": DA300 as a fraction 0-1, white 50 % line;
    kind "ase": hours of direct sun above 1000 lx, cyan 250 h line."""
    import contourpy
    ny, nx = values.shape
    z = radiance_scene.WORKPLANE_Z
    ox, cell, xs = _grid(nx, W)
    oy, _, ys = _grid(ny, D)
    if kind == "da300":
        shown, vmax, colors, level = values * 100, 100, _rgb(values.ravel()), 0.5
        layers, rgb_line = (HEATMAP_LAYER[0], CONTOUR_LAYER[0]), (255, 255, 255)
    else:
        shown, vmax, level = values, ASE_VMAX, ASE_HOURS
        colors = [[int(round(c * 255)) for c in ASE_CMAP(min(float(h) / ASE_VMAX, 1.0))[:3]] for h in values.ravel()]
        layers, rgb_line = (ASE_HEATMAP_LAYER[0], ASE_CONTOUR_LAYER[0]), ASE_CONTOUR_LAYER[1]
    call_rhino("create_heatmap_plane", {
        "ox": ox + dx, "oy": oy, "z": z, "nx": nx, "ny": ny, "dx": cell, "dy": cell,
        "values": shown.ravel().tolist(), "vmin": 0, "vmax": vmax,
        "colors": colors, "smooth": True, "layer": layers[0]})
    if not values.min() < level < values.max():
        return
    for line in contourpy.contour_generator(xs, ys, values).lines(level):
        if np.linalg.norm(np.diff(line, axis=0), axis=1).sum() > 1.0:   # skip loops around single sensors
            call_rhino("create_polyline", {"points": [[x + dx, y, z + 0.01] for x, y in line],
                                           "layer": layers[1], "r": rgb_line[0], "g": rgb_line[1], "b": rgb_line[2]})


def _draw_room(polygons, holes_supported, dx=0.0):
    """Walls, glazing, overhangs, floor and ceiling of one room, shifted by dx along x."""
    n = _draw_walls(polygons, LAYERS["wall_mat"][1], holes_supported, dx)
    for poly in polygons:
        if poly["material"] not in LAYERS or poly["material"] == "wall_mat":   # the 30 m ground is not drawn
            continue
        layer, rgb = LAYERS[poly["material"]]
        call_rhino("create_planar_surface", {"points": _shift(poly["points"], dx), "layer": layer,
                                             "r": rgb[0], "g": rgb[1], "b": rgb[2]})
        n += 1
    return n


def _draw(case, label, da300_map=None, ase_map=None, maps_source="Radiance"):
    """Draw the simulated geometry (from the same room.rad) and the result in Rhino.
    With maps: the room with the DA300 map, and a copy of the room ASE_OFFSET_M to the east
    with the map of direct-sun hours. maps_source names where the maps come from."""
    try:
        requests.get(RHINO_URL, timeout=3)
    except Exception:
        return "Rhino bridge not reachable: in Rhino run ScriptEditor -> 5_mcp/rhino_bridge.py -> Run."
    W, D, H = decision.design_space()["room"]
    scratch = config.MCP_RUNS / "_draw"
    if scratch.exists():
        for f in scratch.iterdir():
            f.unlink()
    room, _, _ = radiance_scene.write_scene(scratch, case, W, D, H)

    all_layers = list(LAYERS.values()) + [HEATMAP_LAYER, CONTOUR_LAYER, ASE_HEATMAP_LAYER, ASE_CONTOUR_LAYER,
                                          LABEL_LAYER]
    for layer, rgb in all_layers:
        call_rhino("create_layer", {"name": layer, "r": rgb[0], "g": rgb[1], "b": rgb[2]})
        # objects on a hidden layer are skipped when deleting, so show the layer first
        call_rhino("set_layer_visible", {"name": layer, "visible": True})
        call_rhino("delete_layer_objects", {"name": layer})
    tools = {t["name"]: t for t in requests.get(f"{RHINO_URL}/tools", timeout=5).json()}
    holes_supported = "holes" in tools.get("create_planar_surface", {}).get("args", {})
    polygons = radiance_scene.read_polygons(room)
    n = _draw_room(polygons, holes_supported)
    drawn = []
    if da300_map is not None:
        _draw_map(np.asarray(da300_map, float), "da300", W, D)
        label += f"\nDA300 map ({maps_source}), white line: 50 %"
        drawn.append("DA300 map")
    call_rhino("create_text_dot", {"x": W / 2, "y": D / 2, "z": H + 1.0, "text": label, "layer": LABEL_LAYER[0]})
    if ase_map is not None:
        n += _draw_room(polygons, holes_supported, ASE_OFFSET_M)
        _draw_map(np.asarray(ase_map, float), "ase", W, D, ASE_OFFSET_M)
        call_rhino("create_text_dot", {"x": ASE_OFFSET_M + W / 2, "y": D / 2, "z": H + 1.0,
                                       "text": f"ASE map ({maps_source})\nhours of direct sun above 1000 lx\n"
                                               f"cyan line: {ASE_HOURS} h", "layer": LABEL_LAYER[0]})
        drawn.append(f"ASE map on a copy of the room {ASE_OFFSET_M:g} m to the east")
    for layer, _ in all_layers:
        call_rhino("set_layer_wire_density", {"name": layer, "density": -1})    # no isocurves on surfaces
    call_rhino("set_layer_visible", {"name": LAYERS["ceiling_mat"][0], "visible": False})
    call_rhino("zoom_extents", {})
    return f"drawn in Rhino: {n} surfaces" + "".join(f", {d}" for d in drawn)


def _label(final, source_note=""):
    return (f"sDA {final['sDA_300_50_percent']:.1f}%  |  ASE {final['ASE_1000_250_percent']:.1f}%\n"
            f"LEED: {final['leed_level']}\nsource: {final['source']}{source_note}")


def _clean(result):
    return json.dumps(result, indent=2, ensure_ascii=False, default=str)


def _with_report(text, result):
    """The fixed report first, then the structured data it was built from."""
    return _clean({"report": text, **result})


# -- workflow tools -------------------------------------------------------------

@mcp.tool()
def design_space_info() -> str:
    """Valid inputs of the daylight model (sampled design space), LEED thresholds, 90% error margins
    and spread-level boundaries. Call this first if unsure which design parameters are allowed."""
    space = decision.design_space()
    cfg = decision.decision_config()
    return _clean({
        "room_m": space["room"], "location": "Milan (ITA_LM_Milano-Linate TMYx)",
        "facade_patterns": ["+".join(p) for p in space["facade_patterns"]],
        "window_types": {"conventional": {"sill_height_m": 0.90,
                                          "window_height_m": space["conventional_window_height"]},
                         "floor_to_ceiling": {"sill_height_m": 0.0, "window_height_m": space["room"][2]}},
        "window_width_m": space["window_width"],
        "overhang_depth_m": f"0 (none) or {space['overhang_depth'][0]}-{space['overhang_depth'][1]}",
        "geometry_rules": "same window on every active façade, centred; overhang at the window head, "
                          "as long as the window is wide",
        "leed_thresholds_percent": decision.LEED_THRESHOLDS,
        "error_margins_pp_90": cfg["margins_pp"], "spread_levels_pp": cfg["spread_levels_pp"],
        "advisory_outputs": {
            "suggestions": "suggest_changes: closest designs of the design space that reach the next LEED level "
                           "robustly (MLP ensemble search, Radiance optional); advisory",
            "maps": "predicted 10 x 10 DA300 map [%] and direct-sun-hour map [h] at 0.3-5.7 m (PCA + MLP) for accepted MLP "
                    "predictions; " + map_prediction.MAP_ACCURACY["da300"] + " (DA300), " + map_prediction.MAP_ACCURACY["ase_hours"]
                    + " (ASE); the decision uses only sDA and ASE",
        },
    })


@mcp.tool()
def evaluate_design(active_facades: list[str], window_type: str, window_width_m: float,
                    window_height_m: Optional[float] = None, sill_height_m: Optional[float] = None,
                    overhang_depth_m: float = 0.0, run_radiance_if_needed: bool = True,
                    draw_in_rhino: bool = True) -> str:
    """Evaluate a design with the thesis decision rule (Section 4.4) and return sDA, ASE, LEED level and source.

    active_facades: windowed façades, e.g. ["south"] or ["east", "south"] (north/east/south/west).
    window_type: "conventional" (sill 0.90 m, give window_height_m) or "floor_to_ceiling" (height 3 m).
    overhang_depth_m: 0 for no overhang. Steps: input coverage check -> mean of five MLPs ->
    if prediction +/- 90% error margin crosses a LEED threshold (sDA 55/75%, ASE 10%) or the design is
    outside the sampled design space, Radiance is run (about 30-60 s) and replaces both metrics.
    Advisory only (never change the decision): spread level, north-only warning and, for an accepted MLP
    prediction, "mlp_maps" (predicted 10 x 10 DA300 and direct-sun-hour maps). Optionally draws the design in Rhino: the room with the DA300 map and,
    8 m to the east, a copy with the direct-sun-hour (ASE) map; predicted or simulated, as labelled.
    The result starts with "report", the fixed report to show to the user unchanged."""
    case = _case(active_facades, window_type, window_width_m, window_height_m, sill_height_m, overhang_depth_m)
    result = decision.evaluate(case)
    if result["decision"] == "invalid":
        return _with_report(report.rejection_report(result["coverage"]["problems"]), result)
    da300 = ase = maps = None
    source = "Radiance"
    if result["needs_radiance"]:
        if run_radiance_if_needed:
            metrics = _run_radiance(case)
            decision.with_radiance(result, metrics)
            result["radiance_output"] = metrics["out_dir"]
            da300, ase = metrics["da300_map"], metrics["ase_hours_map"]
            maps = "simulated"
        else:
            result["final"] = None
            result["note"] = "Radiance needed but not run (run_radiance_if_needed=false); call run_radiance."
    elif "mlp_maps" in result:
        da300 = np.array(result["mlp_maps"]["da300"]["values"]) / 100
        ase = np.array(result["mlp_maps"]["ase_hours"]["values"])
        source = "predicted by PCA + MLP, 10 x 10"
        maps = "predicted"
    if draw_in_rhino:
        label = _label(result["final"]) if result.get("final") else "Radiance simulation required"
        result["rhino"] = _draw(case, label, da300, ase, source)
    return _with_report(report.evaluation_report(result, maps), result)


@mcp.tool()
def run_radiance(active_facades: list[str], window_type: str, window_width_m: float,
                 window_height_m: Optional[float] = None, sill_height_m: Optional[float] = None,
                 overhang_depth_m: float = 0.0, draw_in_rhino: bool = True) -> str:
    """Run the annual Radiance simulation (sDA300/50%, ASE1000,250h) for a design, with the same settings
    as the training data, e.g. to confirm an MLP prediction with a high spread level. Takes about 30-60 s.
    Same design parameters as evaluate_design. Draws the geometry with the DA300 map and, 8 m to the east,
    a copy with the map of direct-sun hours (ASE) in Rhino.
    The result starts with "report", the fixed report to show to the user unchanged."""
    case = _case(active_facades, window_type, window_width_m, window_height_m, sill_height_m, overhang_depth_m)
    coverage = decision.check_coverage(case)
    if not coverage["valid_geometry"]:
        return _with_report(report.rejection_report(coverage["problems"]),
                            {"error": "geometry cannot be built", "problems": coverage["problems"]})
    metrics = _run_radiance(case)
    result = decision.with_radiance({"design": case, "coverage": coverage}, metrics)
    result["radiance_output"] = metrics["out_dir"]
    result["radiance_s"] = round(metrics["total_model_s"], 1)
    if draw_in_rhino:
        result["rhino"] = _draw(case, _label(result["final"]), metrics["da300_map"], metrics["ase_hours_map"])
    return _with_report(report.evaluation_report(result, "simulated"), result)


@mcp.tool()
def draw_design(active_facades: list[str], window_type: str, window_width_m: float,
                window_height_m: Optional[float] = None, sill_height_m: Optional[float] = None,
                overhang_depth_m: float = 0.0) -> str:
    """Draw a design in Rhino (walls, glazing, overhangs; ceiling hidden) without evaluating it."""
    case = _case(active_facades, window_type, window_width_m, window_height_m, sill_height_m, overhang_depth_m)
    coverage = decision.check_coverage(case)
    if not coverage["valid_geometry"]:
        return _clean({"error": "geometry cannot be built", "problems": coverage["problems"]})
    return _draw(case, f"{'+'.join(case['active_facades'])}, {case['window_type']}")


@mcp.tool()
def get_sample_case(model_id: str, draw_in_rhino: bool = True) -> str:
    """Parameters and stored Radiance result of one of the 560 training/test cases (e.g. "model_040"),
    plus what the decision rule gives for it. Draws it in Rhino with its stored DA300 map and, 8 m to the
    east, a copy with its stored map of direct-sun hours (ASE).
    The result starts with "report", the fixed report to show to the user unchanged."""
    cases = {c["model_id"]: c for c in json.loads(config.CASES_JSON.read_text(encoding="utf-8"))["cases"]}
    if model_id not in cases:
        return _clean({"error": f"unknown model_id {model_id} (model_001 ... model_560)"})
    case = cases[model_id]
    result = decision.evaluate(case)
    folder = config.RADIANCE_560 / model_id
    stored = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    result["stored_radiance"] = {k: stored[k] for k in ("sDA_300_50_percent", "ASE_1000_250_percent")}
    result["stored_radiance"]["leed_level"] = decision.leed_level(*result["stored_radiance"].values())
    test_ids = (config.ML_TRAINING / "reduced" / "mlp" / "test_model_ids.csv").read_text().split()[1:]
    result["split"] = "test" if model_id in test_ids else "train"
    if result["needs_radiance"]:     # the stored simulation is the Radiance result the rule asks for
        decision.with_radiance(result, result["stored_radiance"])
        result["final"]["source"] = "stored Radiance result"
    if draw_in_rhino:
        final = {"source": "stored Radiance result", **result["stored_radiance"]}
        result["rhino"] = _draw(case, f"{model_id}\n" + _label(final), np.load(folder / "da300_map.npy"),
                                np.load(folder / "ase_hours_map.npy"), "stored Radiance result")
    return _with_report(report.evaluation_report(result, "stored", name=model_id), result)


@mcp.tool()
def suggest_changes(active_facades: list[str], window_type: str, window_width_m: float,
                    window_height_m: Optional[float] = None, sill_height_m: Optional[float] = None,
                    overhang_depth_m: float = 0.0, n_suggestions: int = 5,
                    confirm_best_with_radiance: bool = False) -> str:
    """Design suggestions (advisory, thesis Section 4.6): the closest designs of the sampled design space
    that reach the next LEED level (2, then 3 points) robustly, i.e. with the whole 90% error range on the
    right side of the thresholds, found by predicting all 430 976 designs on a 0.05 m grid with the MLP
    ensemble (no Radiance; first call about 10 s, later calls under 1 s). Ranked by a change cost from the
    given design (window size and overhang changes scaled by their range; +1 per change of window type or
    per façade whose window is added/removed). If no design reaches the target, the designs with ASE
    robustly <= 10% and the highest sDA are listed. Optionally the first suggestion is simulated in
    Radiance (about 30 s). Same design parameters as evaluate_design.
    The result starts with "report", the fixed report to show to the user unchanged."""
    import suggest
    case = _case(active_facades, window_type, window_width_m, window_height_m, sill_height_m, overhang_depth_m)
    coverage = decision.check_coverage(case)
    if not coverage["valid_geometry"]:
        return _with_report(report.rejection_report(coverage["problems"]),
                            {"error": "geometry cannot be built", "problems": coverage["problems"]})
    result = {"design": case, "coverage": coverage}
    level_now = None
    if coverage["in_design_space"]:
        ev = decision.evaluate(case)
        source = ev["final"] if ev.get("final") else {
            k: round(ev["mlp"][k]["prediction"], 1) for k in ("sDA_300_50_percent", "ASE_1000_250_percent")}
        result["current"] = {**source, "note": ev["reason"]}
        level_now = suggest._level(ev["mlp"]["sDA_300_50_percent"]["prediction"],
                                   ev["mlp"]["ASE_1000_250_percent"]["prediction"])
    else:
        result["current"] = {"note": "outside the sampled design space: no prediction; target set to 2 points. "
                                     "Use run_radiance for the value of this design."}
        level_now = 0
    result.update(suggest.suggest(case, n=max(1, n_suggestions), level_now=level_now))
    if confirm_best_with_radiance and result.get("suggestions"):
        s = result["suggestions"][0]["design"]
        best = _case(s["active_facades"], s["window_type"], s["window_width_m"],
                     s["window_height_m"] if s["window_type"] == "conventional" else None, None, s["overhang_depth_m"])
        metrics = _run_radiance(best)
        result["radiance_check_of_first_suggestion"] = {
            "sDA_300_50_percent": round(metrics["sDA_300_50_percent"], 1),
            "ASE_1000_250_percent": round(metrics["ASE_1000_250_percent"], 1),
            "leed_level": decision.leed_level(metrics["sDA_300_50_percent"], metrics["ASE_1000_250_percent"]),
            "radiance_output": metrics["out_dir"]}
    return _with_report(report.suggestion_report(result), result)


# -- general Rhino tools ----------------------------------------------------------

@mcp.tool()
def create_sphere(x: float, y: float, z: float, radius: float,
                  layer: Optional[str] = None,
                  r: Optional[int] = None, g: Optional[int] = None, b: Optional[int] = None) -> str:
    """Create a sphere in Rhino. Optionally assign to a layer and set RGB color."""
    args = {"x": x, "y": y, "z": z, "radius": radius}
    if layer: args["layer"] = layer
    if r is not None: args["r"] = r
    if g is not None: args["g"] = g
    if b is not None: args["b"] = b
    return str(call_rhino("create_sphere", args))


@mcp.tool()
def create_box(x: float, y: float, z: float,
               x0: float = 0, y0: float = 0, z0: float = 0,
               layer: Optional[str] = None,
               r: Optional[int] = None, g: Optional[int] = None, b: Optional[int] = None) -> str:
    """Create a box in Rhino from corner (x0,y0,z0) to corner (x,y,z). Optionally assign layer and color."""
    args = {"x0": x0, "y0": y0, "z0": z0, "x": x, "y": y, "z": z}
    if layer: args["layer"] = layer
    if r is not None: args["r"] = r
    if g is not None: args["g"] = g
    if b is not None: args["b"] = b
    return str(call_rhino("create_box", args))


@mcp.tool()
def create_triangle(x1: float, y1: float, z1: float,
                    x2: float, y2: float, z2: float,
                    x3: float, y3: float, z3: float,
                    layer: Optional[str] = None,
                    r: Optional[int] = None, g: Optional[int] = None, b: Optional[int] = None) -> str:
    """Create a triangle mesh face in Rhino. Optionally assign layer and color."""
    args = {"x1": x1, "y1": y1, "z1": z1,
            "x2": x2, "y2": y2, "z2": z2,
            "x3": x3, "y3": y3, "z3": z3}
    if layer: args["layer"] = layer
    if r is not None: args["r"] = r
    if g is not None: args["g"] = g
    if b is not None: args["b"] = b
    return str(call_rhino("create_triangle", args))


@mcp.tool()
def create_layer(name: str, r: int, g: int, b: int) -> str:
    """Create or update a Rhino layer with the given name and RGB color."""
    return str(call_rhino("create_layer", {"name": name, "r": r, "g": g, "b": b}))


@mcp.tool()
def set_object_color(id: str, r: int, g: int, b: int) -> str:
    """Set the display color of a Rhino object by its GUID."""
    return str(call_rhino("set_object_color", {"id": id, "r": r, "g": g, "b": b}))


@mcp.tool()
def delete_object(id: str) -> str:
    """Delete a specific object in Rhino by its ID."""
    return str(call_rhino("delete_object", {"id": id}))


@mcp.tool()
def delete_all_objects() -> str:
    """Delete all objects in the Rhino document."""
    return str(call_rhino("delete_all_objects", {}))


@mcp.tool()
def rhino_status() -> str:
    """Check whether Rhino bridge is running."""
    response = requests.get(RHINO_URL, timeout=5)
    return response.text


@mcp.tool()
def list_rhino_tools() -> str:
    """List all tools currently available on the Rhino bridge, including any newly added ones."""
    response = requests.get(f"{RHINO_URL}/tools", timeout=5)
    response.raise_for_status()
    return response.text


@mcp.tool()
def call_rhino_tool(tool_name: str, args: dict) -> str:
    """Call any Rhino bridge tool by name. Use list_rhino_tools() first to discover available tools and their required args."""
    return str(call_rhino(tool_name, args))


if __name__ == "__main__":
    mcp.run()
