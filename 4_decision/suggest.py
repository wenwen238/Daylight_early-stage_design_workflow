"""Design suggestions by surrogate search (thesis Section 4.6). Advisory; never changes a decision.

suggest(case, n=5)
    Target: the next LEED level above that of the design (2 points, then 3 points).
    Every design of the sampled design space on a 0.05 m grid (14 façade patterns x 32 window
    type/height options x 37 widths x 26 overhang depths = 430 976 designs) is predicted by the
    five MLPs. A design meets the target robustly when its whole 90 % error range lies on the
    right side of the thresholds, i.e. the decision rule would accept it without Radiance:
        sDA - margin_sDA >= 55 (or 75)   and   ASE + margin_ASE <= 10
    These designs are ranked by the change cost from the evaluated design
        |dw| / 1.80 + |dh| / 1.50 + |dd| / 1.50 + 1 per change of window type
        (height term dropped then) + 1 per façade whose window is added or removed,
    and the n cheapest are returned. If none meets the target, the designs with ASE robustly
    <= 10 % and the highest sDA are returned instead (within 2 pp of the best, cheapest first)."""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3_ml"))
import common  # noqa: E402
import decision  # noqa: E402

SDA, ASE = common.TARGETS
STEP = 0.05
FTC_HEIGHT = 3.00
SHORT = {"north": "N", "east": "E", "south": "S", "west": "W"}


def _grid(lo, hi):
    return np.round(np.arange(lo, hi + STEP / 2, STEP), 2)


def _level(sda, ase):
    return 0 if ase > 10.0 else 3 if sda >= 75.0 else 2 if sda >= 55.0 else 0


@lru_cache(maxsize=1)
def _space():
    """All designs of the sampled design space on the 0.05 m grid, with their predictions."""
    sp = decision.design_space()
    widths, heights = _grid(*sp["window_width"]), _grid(*sp["conventional_window_height"])
    overs = np.concatenate([[0.0], _grid(*sp["overhang_depth"])])
    types = [("conventional", h) for h in heights] + [("floor_to_ceiling", FTC_HEIGHT)]
    rows = [(pat, t, w, h, o) for pat in sp["facade_patterns"] for t, h in types for w in widths for o in overs]
    d = pd.DataFrame(rows, columns=["facades", "window_type", "width", "height", "overhang"])
    for f, flag in zip(common.FACADES, common.FACADE_FLAGS):
        d[flag] = d["facades"].apply(lambda fs, f=f: int(f in fs))
    d["sill_height_m"] = np.where(d["window_type"] == "conventional", decision.CONVENTIONAL_SILL_M, 0.0)
    d["window_width_m"], d["window_height_m"], d["overhang_depth_m"] = d["width"], d["height"], d["overhang"]
    X = d[common.feature_columns("reduced")]
    for t, members in decision._models().items():
        p = np.array([common.clip(m.predict(X)) for m in members])
        d[t], d[f"spread_{t}"] = p.mean(axis=0), p.std(axis=0, ddof=1)
    return d


def _cost(d, base):
    same_type = d["window_type"] == base["window_type"]
    height = np.where(same_type, (d["height"] - base["window_height_m"]).abs() / 1.50, 1.0)
    facades = d["facades"].apply(lambda fs: len(set(fs) ^ set(base["active_facades"])))
    return ((d["width"] - base["window_width_m"]).abs() / 1.80 + height
            + (d["overhang"] - base["overhang_depth_m"]).abs() / 1.50 + facades)


def _describe(r, base):
    parts = []
    added = [f for f in r["facades"] if f not in base["active_facades"]]
    removed = [f for f in base["active_facades"] if f not in r["facades"]]
    parts += [f"add a window on the {f} façade" for f in added]
    parts += [f"remove the window on the {f} façade" for f in removed]
    if r["window_type"] != base["window_type"]:
        parts.append(f"change the window type to {r['window_type'].replace('_', '-')}")
    elif abs(r["height"] - base["window_height_m"]) > 1e-6:
        parts.append(f"window height {base['window_height_m']:.2f} -> {r['height']:.2f} m")
    if abs(r["width"] - base["window_width_m"]) > 1e-6:
        parts.append(f"window width {base['window_width_m']:.2f} -> {r['width']:.2f} m")
    if abs(r["overhang"] - base["overhang_depth_m"]) > 1e-6:
        parts.append(f"overhang depth {base['overhang_depth_m']:.2f} -> {r['overhang']:.2f} m")
    return "; ".join(parts) or "no change"


def suggest(case, n=5, level_now=None):
    """Closest designs of the design space that robustly reach the next LEED level."""
    margins = decision.decision_config()["margins_pp"]
    if level_now is None:
        p = decision.predict(case)
        level_now = _level(p[SDA]["mean"], p[ASE]["mean"])
    if level_now >= 3:
        return {"target": None, "note": "the design already reaches 3 points; no suggestion needed"}
    target = 2 if level_now < 2 else 3
    thr = 55.0 if target == 2 else 75.0
    d = _space()
    ok_ase = d[ASE] + margins[ASE] <= 10.0
    hit = d[ok_ase & (d[SDA] - margins[SDA] >= thr)]
    out = {"target": f"{target} points (sDA >= {thr:.0f}%, ASE <= 10%), robust: whole 90% error range "
                     f"on the right side of the thresholds",
           "designs_searched": int(len(d)), "designs_reaching_target": int(len(hit))}
    if len(hit):
        pick = hit.assign(cost=_cost(hit, case)).sort_values("cost").head(n)
    else:
        cand = d[ok_ase]
        best = float(cand[SDA].max())
        pick = cand[cand[SDA] >= best - 2.0]
        pick = pick.assign(cost=_cost(pick, case)).sort_values("cost").head(n)
        out["note"] = (f"no design in the sampled design space reaches the target robustly. Listed instead: "
                       f"designs with ASE robustly <= 10% and the highest predicted sDA (max {best:.1f}%)")
    out["suggestions"] = [{
        "change": _describe(r, case),
        "design": {"active_facades": list(r["facades"]), "window_type": r["window_type"],
                   "window_width_m": float(r["width"]),
                   "window_height_m": float(r["height"]), "overhang_depth_m": float(r["overhang"])},
        "change_cost": round(float(r["cost"]), 2),
        "predicted_sDA_300_50_percent": round(float(r[SDA]), 1),
        "predicted_ASE_1000_250_percent": round(float(r[ASE]), 1),
        "spread_level": {"sDA": decision.spread_level(SDA, r[f"spread_{SDA}"]),
                         "ASE": decision.spread_level(ASE, r[f"spread_{ASE}"])},
        "leed_level": decision.leed_level(r[SDA], r[ASE]),
    } for _, r in pick.iterrows()]
    out["advisory"] = ("predictions of the MLP ensemble within the sampled design space and the fixed reference "
                       "room; within the range the decision rule accepts, so Radiance is optional. Feasibility in "
                       "the real building (structure, heritage, glare, energy, cost) is not considered.")
    return out
