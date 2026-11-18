#!/usr/bin/env python3
"""Decision rule between the MLP ensemble prediction and a Radiance simulation (thesis Section 4.4).

1. Input coverage check (hard gate). The design must lie in the sampled design space:
   a façade pattern present in the training data, the window-type rules
   (conventional: sill 0.90 m, height 0.60-2.10 m; floor-to-ceiling: sill 0 m,
   height 3.00 m), window width 0.60-2.40 m, overhang depth 0 m or 0.30-1.50 m.
   The fixed study conditions (room, climate, materials, schedule, sensor grid,
   no obstructions) cannot be changed through this module, so they always hold.
   Outside the design space -> Radiance. Geometry that cannot exist (e.g. a window
   wider than the façade) is rejected.
2. Ensemble prediction: mean of five MLPs (seeds 20260924-20260928), each clipped to 0-100 %.
3. Threshold check: prediction +/- 90 % empirical error margin (sDA, ASE) is compared
   with the LEED thresholds (sDA 55 % and 75 %, ASE 10 %). If a range contains a
   threshold -> Radiance for both metrics; otherwise the MLP prediction is accepted.
4. Advisory only (never changes the decision): spread level low/medium/high from the
   SD of the five predictions; warning for north-only designs (ASE). For an accepted
   MLP prediction also the predicted 10 x 10 DA300 and direct-sun-hour maps (map_prediction.py).

The margins and spread-level boundaries are derived from the 112 test cases by
`python decision.py calibrate` and stored in decision_config.json.

Command line
------------
    python 4_decision/decision.py calibrate        # margins and spread levels from the test set
    python 4_decision/decision.py check-test       # apply the rule to the 112 test cases
    python 4_decision/decision.py evaluate --facades south --window-type conventional \
        --width 1.8 --height 1.5 --overhang 0.6
"""

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3_ml"))
import config  # noqa: E402
import common  # noqa: E402
import map_prediction  # noqa: E402

SEEDS = [20260924, 20260925, 20260926, 20260927, 20260928]   # as train_mlp_ensemble.py
COVERAGE_LEVEL = 0.90
LEED_THRESHOLDS = {"sDA_300_50_percent": (55.0, 75.0), "ASE_1000_250_percent": (10.0,)}
SHORT = {"sDA_300_50_percent": "sDA", "ASE_1000_250_percent": "ASE"}
FACADE_ORDER = common.FACADES                                  # north, east, south, west
CONVENTIONAL_SILL_M = 0.90
TOL = 1e-6


# -- design space -----------------------------------------------------------

@lru_cache(maxsize=1)
def design_space():
    """Sampled design space, read from the fixed 560-case file."""
    data = json.loads(config.CASES_JSON.read_text(encoding="utf-8"))
    meta = data["metadata"]
    patterns = sorted({tuple(sorted(c["active_facades"], key=FACADE_ORDER.index)) for c in data["cases"]})
    room = meta["room_dimensions_m"]
    ranges = meta["continuous_ranges_m"]
    return {
        "room": (room["width"], room["depth"], room["height"]),
        "facade_patterns": patterns,
        "window_width": tuple(ranges["window_width"]),
        "conventional_window_height": tuple(ranges["conventional_window_height"]),
        "overhang_depth": tuple(ranges["overhang_depth_when_present"]),
    }


def make_case(active_facades, window_type, window_width_m, window_height_m=None,
              sill_height_m=None, overhang_depth_m=0.0, model_id="design"):
    """Build a design case in the format of fixed_design_cases_560.json.

    Same geometric rules as the sampling: the same window on every active façade,
    centred horizontally; the overhang (if any) sits at the window head and is as
    long as the window is wide.
    """
    W, D, H = design_space()["room"]
    facades = sorted({f.strip().lower() for f in active_facades}, key=lambda f: FACADE_ORDER.index(f)
                     if f in FACADE_ORDER else 99)
    window_type = window_type.strip().lower().replace("-", "_").replace(" ", "_")
    if sill_height_m is None:
        sill_height_m = 0.0 if window_type == "floor_to_ceiling" else CONVENTIONAL_SILL_M
    if window_height_m is None and window_type == "floor_to_ceiling":
        window_height_m = H - sill_height_m
    width, height, sill = float(window_width_m), float(window_height_m), float(sill_height_m)
    depth = float(overhang_depth_m or 0.0)
    x0 = round((W - width) / 2, 4)
    x1 = round(x0 + width, 4)
    z0, z1 = round(sill, 4), round(sill + height, 4)
    windows = [{"face": f, "x0_m": x0, "x1_m": x1, "z0_m": z0, "z1_m": z1} for f in facades]
    shading = [{"type": "overhang", "face": f, "x0_m": x0, "x1_m": x1, "z_m": z1, "depth_m": depth}
               for f in facades] if depth > 0 else []
    return {
        "model_id": model_id,
        "active_facades": facades,
        "window_type": window_type,
        "sill_height_m": sill,
        "window_width_m": width,
        "window_height_m": height,
        "overhang_present": depth > 0,
        "overhang_depth_m": depth,
        "overhang_length_m": width if depth > 0 else 0.0,
        "windows": windows,
        "shading": shading,
    }


def _within(value, lo, hi):
    return lo - TOL <= value <= hi + TOL


def check_coverage(case):
    """Return {"valid_geometry", "in_design_space", "problems", "outside"}."""
    space = design_space()
    W, D, H = space["room"]
    facades = sorted(case["active_facades"], key=lambda f: FACADE_ORDER.index(f) if f in FACADE_ORDER else 99)
    width, height = case["window_width_m"], case["window_height_m"]
    sill, depth = case["sill_height_m"], case["overhang_depth_m"]

    problems = []        # geometry that cannot be built
    unknown = [f for f in facades if f not in FACADE_ORDER]
    if unknown:
        problems.append(f"unknown façade name(s): {unknown} (use north, east, south, west)")
    if not facades:
        problems.append("no windowed façade given")
    if case["window_type"] not in ("conventional", "floor_to_ceiling"):
        problems.append(f"unknown window type '{case['window_type']}' (use conventional or floor_to_ceiling)")
    if width <= 0 or height <= 0:
        problems.append("window width and height must be positive")
    if width > min(W, D) + TOL:
        problems.append(f"window width {width} m is wider than the façade ({min(W, D)} m)")
    if sill < -TOL or sill + height > H + TOL:
        problems.append(f"window (sill {sill} m + height {height} m) does not fit the {H} m façade")
    if depth < 0:
        problems.append("overhang depth cannot be negative")

    outside = []         # buildable, but not represented in the training data
    if not problems:
        if tuple(facades) not in space["facade_patterns"]:
            outside.append(f"façade pattern {'+'.join(facades)} is not in the training data "
                           "(all four façades were never sampled)")
        if case["window_type"] == "conventional":
            if abs(sill - CONVENTIONAL_SILL_M) > TOL:
                outside.append(f"conventional window sill {sill} m (trained only with {CONVENTIONAL_SILL_M} m)")
            lo, hi = space["conventional_window_height"]
            if not _within(height, lo, hi):
                outside.append(f"window height {height} m outside {lo}-{hi} m")
        else:
            if abs(sill) > TOL or abs(height - H) > TOL:
                outside.append(f"floor-to-ceiling window must have sill 0 m and height {H} m")
        lo, hi = space["window_width"]
        if not _within(width, lo, hi):
            outside.append(f"window width {width} m outside {lo}-{hi} m")
        lo, hi = space["overhang_depth"]
        if depth > TOL and not _within(depth, lo, hi):
            outside.append(f"overhang depth {depth} m outside 0 m or {lo}-{hi} m")
    return {"valid_geometry": not problems, "in_design_space": not problems and not outside,
            "problems": problems, "outside": outside}


# -- prediction -------------------------------------------------------------

@lru_cache(maxsize=1)
def _models():
    return {t: [joblib.load(config.ENSEMBLE_DIR / f"model_{t}_seed{s}.joblib") for s in SEEDS]
            for t in common.TARGETS}


def features(case):
    row = {flag: int(f in case["active_facades"]) for f, flag in zip(common.FACADES, common.FACADE_FLAGS)}
    row.update({k: float(case[k]) for k in common.NUMERICAL})
    return pd.DataFrame([row])[common.feature_columns("reduced")]


def predict(case):
    """Ensemble mean and spread (SD, ddof=1) of the five clipped member predictions."""
    X = features(case)
    out = {}
    for t, members in _models().items():
        p = np.array([common.clip(m.predict(X))[0] for m in members])
        out[t] = {"members": p.round(3).tolist(), "mean": float(p.mean()), "spread": float(p.std(ddof=1))}
    return out


# -- calibration --------------------------------------------------------------

def calibrate(write=True):
    """90 % error margins and spread-level boundaries from the 112 test cases."""
    d = pd.read_csv(config.ENSEMBLE_DIR / "ensemble_test_predictions.csv", dtype={"model_id": str})
    cfg = {"coverage_level": COVERAGE_LEVEL, "n_test_cases": len(d), "margins_pp": {}, "spread_levels_pp": {},
           "source": "thesis_data/results/ml_training/reduced/mlp_ensemble/ensemble_test_predictions.csv"}
    for t in common.TARGETS:
        err = (d[f"mean_{t}"] - d[f"actual_{t}"]).abs()
        cfg["margins_pp"][t] = float(np.quantile(err, COVERAGE_LEVEL))
        # three equal-size groups by spread, as in the thesis figure (thesis_data/figure_scripts/fig_ensemble_uncertainty.py)
        spread = d[f"std_{t}"]
        group = pd.qcut(spread.rank(method="first"), 3, labels=False)
        medium = spread[group == 1]
        cfg["spread_levels_pp"][t] = {"low_below": float(medium.min()), "high_above": float(medium.max()),
                                      "group_MAE_pp": [float(err[group == g].mean()) for g in range(3)]}
    if write:
        config.DECISION_CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


@lru_cache(maxsize=1)
def decision_config():
    if not config.DECISION_CONFIG.exists():
        return calibrate(write=True)
    return json.loads(config.DECISION_CONFIG.read_text(encoding="utf-8"))


def spread_level(target, spread):
    b = decision_config()["spread_levels_pp"][target]
    if spread < b["low_below"]:
        return "low"
    if spread > b["high_above"]:
        return "high"
    return "medium"


def leed_level(sda, ase):
    """LEED v4 BD+C EQ Daylight Option 1 benchmark (comparative, not certification)."""
    if ase > 10.0:
        return "not met (ASE > 10%)"
    if sda >= 75.0:
        return "3 points (sDA >= 75%, ASE <= 10%)"
    if sda >= 55.0:
        return "2 points (sDA >= 55%, ASE <= 10%)"
    return "not met (sDA < 55%)"


def threshold_check(pred):
    """Which thresholds lie inside prediction +/- margin (strictly closer than the margin)."""
    margins = decision_config()["margins_pp"]
    crossed = {}
    for t in common.TARGETS:
        m = pred[t]["mean"]
        crossed[t] = [thr for thr in LEED_THRESHOLDS[t] if abs(m - thr) < margins[t]]
    return crossed


# -- the rule -----------------------------------------------------------------

def evaluate(case):
    """Apply the selection rule. Returns what to report and whether Radiance is needed."""
    coverage = check_coverage(case)
    result = {"design": {k: case[k] for k in ("active_facades", "window_type", "sill_height_m", "window_width_m",
                                               "window_height_m", "overhang_depth_m")},
              "coverage": coverage, "warnings": []}
    if not coverage["valid_geometry"]:
        result.update(decision="invalid", needs_radiance=False,
                      reason="geometry cannot be built: " + "; ".join(coverage["problems"]))
        return result
    if not coverage["in_design_space"]:
        result.update(decision="radiance", needs_radiance=True,
                      reason="outside the sampled design space: " + "; ".join(coverage["outside"]))
        return result

    pred = predict(case)
    margins = decision_config()["margins_pp"]
    crossed = threshold_check(pred)
    result["mlp"] = {
        t: {"prediction": round(pred[t]["mean"], 2), "margin_pp": round(margins[t], 2),
            "range": [round(pred[t]["mean"] - margins[t], 2), round(pred[t]["mean"] + margins[t], 2)],
            "thresholds_in_range": crossed[t], "spread_pp": round(pred[t]["spread"], 3),
            "spread_level": spread_level(t, pred[t]["spread"]), "members": pred[t]["members"]}
        for t in common.TARGETS}
    if set(case["active_facades"]) == {"north"}:
        result["warnings"].append(
            "north-only design: all 40 north-only training cases had a simulated ASE of 0%, "
            "while the MLP can predict small non-zero ASE values; the prediction is not forced to 0%")
    for t in common.TARGETS:
        if result["mlp"][t]["spread_level"] == "high":
            result["warnings"].append(f"high uncertainty for {SHORT[t]} (high spread level): "
                                      "you may want to confirm it with a Radiance simulation")

    flagged = [f"{SHORT[t]} {pred[t]['mean']:.1f}% +/- {margins[t]:.1f} contains {crossed[t]}"
               for t in common.TARGETS if crossed[t]]
    if flagged:
        result.update(decision="radiance", needs_radiance=True,
                      reason="the LEED outcome is uncertain: " + "; ".join(flagged))
    else:
        sda, ase = pred["sDA_300_50_percent"]["mean"], pred["ASE_1000_250_percent"]["mean"]
        result.update(decision="mlp", needs_radiance=False,
                      reason="no LEED threshold within the 90% error margins",
                      final={"source": "MLP ensemble", "sDA_300_50_percent": round(sda, 1),
                             "ASE_1000_250_percent": round(ase, 1), "leed_level": leed_level(sda, ase)})
        result["mlp_maps"] = map_prediction.predict_maps(features(case))       # advisory, see map_prediction.py
    return result


def with_radiance(result, metrics):
    """Replace both metrics by the Radiance result (one simulation gives sDA and ASE)."""
    sda, ase = metrics["sDA_300_50_percent"], metrics["ASE_1000_250_percent"]
    result["final"] = {"source": "Radiance simulation", "sDA_300_50_percent": round(sda, 1),
                       "ASE_1000_250_percent": round(ase, 1), "leed_level": leed_level(sda, ase)}
    return result


# -- command line ---------------------------------------------------------------

def check_test():
    """Apply the rule to the 112 test cases (thesis Section 4.5.3 numbers)."""
    d = pd.read_csv(config.ENSEMBLE_DIR / "ensemble_test_predictions.csv", dtype={"model_id": str})
    cases = {c["model_id"]: c for c in json.loads(config.CASES_JSON.read_text(encoding="utf-8"))["cases"]}
    n_cov, flagged, by = 0, [], {t: 0 for t in common.TARGETS}
    agree = 0
    for _, row in d.iterrows():
        case = cases[row.model_id]
        r = evaluate(case)
        n_cov += r["coverage"]["in_design_space"]
        # the live prediction must equal the stored ensemble mean
        for t in common.TARGETS:
            assert abs(r["mlp"][t]["prediction"] - row[f"mean_{t}"]) < 0.01, (row.model_id, t)
            by[t] += bool(r["mlp"][t]["thresholds_in_range"])
        if r["needs_radiance"]:
            flagged.append(row.model_id)
        pred_level = leed_level(r["mlp"]["sDA_300_50_percent"]["prediction"], r["mlp"]["ASE_1000_250_percent"]["prediction"])
        agree += pred_level == leed_level(row["actual_sDA_300_50_percent"], row["actual_ASE_1000_250_percent"])
    cfg = decision_config()
    print(f"margins (90%): sDA {cfg['margins_pp']['sDA_300_50_percent']:.3f} pp, "
          f"ASE {cfg['margins_pp']['ASE_1000_250_percent']:.3f} pp")
    for t in common.TARGETS:
        s = cfg["spread_levels_pp"][t]
        print(f"spread levels {SHORT[t]}: low < {s['low_below']:.3f} <= medium <= {s['high_above']:.3f} < high; "
              f"group MAE {[round(v, 2) for v in s['group_MAE_pp']]}")
    print(f"test cases: {len(d)}, passed coverage check: {n_cov}")
    print(f"sent to Radiance: {len(flagged)} (sDA {by['sDA_300_50_percent']}, ASE {by['ASE_1000_250_percent']}, "
          f"both {by['sDA_300_50_percent'] + by['ASE_1000_250_percent'] - len(flagged)})")
    print(f"LEED level from ensemble mean = Radiance: {agree}/{len(d)}")
    print("flagged:", " ".join(flagged))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calibrate")
    sub.add_parser("check-test")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--facades", nargs="+", required=True)
    ev.add_argument("--window-type", required=True, choices=["conventional", "floor_to_ceiling"])
    ev.add_argument("--width", type=float, required=True)
    ev.add_argument("--height", type=float)
    ev.add_argument("--sill", type=float)
    ev.add_argument("--overhang", type=float, default=0.0)
    args = parser.parse_args()
    if args.cmd == "calibrate":
        print(json.dumps(calibrate(), indent=2))
    elif args.cmd == "check-test":
        check_test()
    else:
        case = make_case(args.facades, args.window_type, args.width, args.height, args.sill, args.overhang)
        print(json.dumps(evaluate(case), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
