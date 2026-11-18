#!/usr/bin/env python3
"""Independent check of the surrogate and the decision rule on designs outside the 560 cases.

Three groups, fixed before any simulation is run (test_protocol.json, designs_and_predictions_before_simulation.json):
  A  100 random designs of the sampled design space: accuracy and coverage of the 90 % margins
  B   30 designs near a LEED threshold: 5 inside and 5 just outside the margin (up to 2 x margin)
      for each of sDA 55 %, sDA 75 % and ASE 10 %: can the rule catch a wrong side of a threshold?
  C   20 first suggestions of suggest.py from different starting designs: selection bias
Only group A estimates the general coverage; B and C are selected on purpose.
The results do not change the models, the margins or the rule: nothing in the workflow is modified.

Usage:  python 6_extensions/independent_validation.py --prepare
        python 6_extensions/independent_validation.py --run --workers 8
        python 6_extensions/independent_validation.py --analyse
Output: thesis_data/results/extensions/independent_validation/
The Radiance run folders (runs/<id>/) are not kept in the repository; without them --analyse reads
the simulated sDA / ASE from the saved predicted_vs_radiance_147_designs.csv.
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, spearmanr

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "3_ml", ROOT / "4_decision"):
    sys.path.insert(0, str(p))
import config  # noqa: E402
import common  # noqa: E402
import decision  # noqa: E402
import suggest  # noqa: E402

OUT = config.RESULTS / "extensions" / "independent_validation"
RUNS = OUT / "runs"
SEED = 20261104                        # not used anywhere in the 560 cases or the training
POOL = 200_000
N_A, N_B_CELL, N_C = 100, 5, 20
SDA, ASE = common.TARGETS
SHORT = {SDA: "sDA", ASE: "ASE"}
THRESHOLDS = [(SDA, 55.0), (SDA, 75.0), (ASE, 10.0)]
BLUE, INK, INK2, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 9.5, "axes.titlecolor": INK, "axes.linewidth": 0.8,
})


# -- designs ----------------------------------------------------------------------

def strata():
    return [(p, t, o) for p in decision.design_space()["facade_patterns"]
            for t in ("conventional", "floor_to_ceiling") for o in (False, True)]


def lhs(rng, low, high, n):
    return low + (high - low) * rng.permutation((np.arange(n) + rng.random(n)) / n)


def table(rng, chosen):
    """Designs for the given strata, with Latin hypercube values of the continuous inputs."""
    sp = decision.design_space()
    n = len(chosen)
    width = np.round(lhs(rng, *sp["window_width"], n), 2)
    height = np.round(lhs(rng, *sp["conventional_window_height"], n), 2)
    depth = np.round(lhs(rng, *sp["overhang_depth"], n), 2)
    rows = []
    for i, (pattern, kind, over) in enumerate(chosen):
        conventional = kind == "conventional"
        rows.append({"facades": pattern, "window_type": kind, "window_width_m": float(width[i]),
                     "window_height_m": float(height[i]) if conventional else 3.0,
                     "sill_height_m": decision.CONVENTIONAL_SILL_M if conventional else 0.0,
                     "overhang_depth_m": float(depth[i]) if over else 0.0})
    return pd.DataFrame(rows)


def predict(d):
    d = d.copy()
    for f, flag in zip(common.FACADES, common.FACADE_FLAGS):
        d[flag] = d["facades"].apply(lambda fs, f=f: int(f in fs))
    X = d[common.feature_columns("reduced")]
    for t, members in decision._models().items():
        p = np.array([common.clip(m.predict(X)) for m in members])
        d[f"pred_{t}"], d[f"spread_{t}"] = p.mean(axis=0), p.std(axis=0, ddof=1)
    return d


def key(row):
    return (tuple(row["facades"]), row["window_type"], round(row["window_width_m"], 2),
            round(row["window_height_m"], 2), round(row["overhang_depth_m"], 2))


def known_keys():
    cases = json.loads(config.CASES_JSON.read_text(encoding="utf-8"))["cases"]
    return {(tuple(sorted(c["active_facades"], key=common.FACADES.index)), c["window_type"],
             round(c["window_width_m"], 2), round(c["window_height_m"], 2), round(c["overhang_depth_m"], 2))
            for c in cases}


def group_a(rng):
    s = strata()
    extra = rng.choice(len(s), N_A - len(s), replace=False)
    chosen = s + [s[i] for i in extra]
    return predict(table(rng, [chosen[i] for i in rng.permutation(len(chosen))]))


def random_pool(rng):
    s = strata()
    return predict(table(rng, [s[i] for i in rng.integers(len(s), size=POOL)]))


def group_b(pool, rng):
    margins = decision.decision_config()["margins_pp"]
    dist = {(t, thr): (pool[f"pred_{t}"] - thr).abs() for t, thr in THRESHOLDS}
    crossed = pd.DataFrame({k: v < margins[k[0]] for k, v in dist.items()})
    # the other metric should not settle the LEED outcome by itself, so that this threshold decides
    decisive = {(SDA, 55.0): pool[f"pred_{ASE}"] + margins[ASE] <= 10.0,
                (SDA, 75.0): pool[f"pred_{ASE}"] + margins[ASE] <= 10.0,
                (ASE, 10.0): pool[f"pred_{SDA}"] - margins[SDA] >= 55.0}
    picks, log = [], []
    for t, thr in THRESHOLDS:
        d, m = dist[(t, thr)], margins[t]
        others = crossed.drop(columns=[(t, thr)]).any(axis=1)
        cells = {"inside": (d < m) & ~others, "outside": (d >= m) & (d < 2 * m) & ~others}
        for cell, mask in cells.items():
            strict = pool[mask & decisive[(t, thr)]]
            use_strict = len(strict) >= N_B_CELL
            cand = strict if use_strict else pool[mask]
            log.append({"threshold": f"{SHORT[t]} {thr:g}%", "cell": cell, "available_decisive": len(strict),
                        "available_any": int(mask.sum()), "used": "decisive" if use_strict else "any"})
            take = cand.iloc[rng.choice(len(cand), N_B_CELL, replace=False)].copy()
            take["threshold_metric"], take["threshold_value"], take["cell"] = t, thr, cell
            take["other_metric_decisive"] = use_strict
            picks.append(take)
    return pd.concat(picks, ignore_index=True), log


def group_c(pool, rng):
    rows, skipped, seen, tried = [], 0, set(), 0
    order = rng.permutation(len(pool))
    for i in order:
        if len(rows) == N_C:
            break
        start = pool.iloc[int(i)]
        tried += 1
        level = suggest._level(start[f"pred_{SDA}"], start[f"pred_{ASE}"])
        if level >= 3:
            continue
        case = decision.make_case(list(start["facades"]), start["window_type"], start["window_width_m"],
                                  start["window_height_m"], start["sill_height_m"], start["overhang_depth_m"])
        out = suggest.suggest(case, n=1, level_now=level)
        if out.get("note") or not out.get("suggestions"):     # no design reaches the target robustly
            skipped += 1
            continue
        s = out["suggestions"][0]["design"]
        row = {"facades": tuple(s["active_facades"]), "window_type": s["window_type"],
               "window_width_m": s["window_width_m"], "window_height_m": s["window_height_m"],
               "sill_height_m": decision.CONVENTIONAL_SILL_M if s["window_type"] == "conventional" else 0.0,
               "overhang_depth_m": s["overhang_depth_m"], "target_points": 2 if level < 2 else 3,
               "change_cost": out["suggestions"][0]["change_cost"], "designs_reaching_target": out["designs_reaching_target"],
               "start_facades": "+".join(start["facades"]), "start_type": start["window_type"],
               "start_width": start["window_width_m"], "start_height": start["window_height_m"],
               "start_overhang": start["overhang_depth_m"], "start_level": level}
        if key(row) in seen:
            continue
        seen.add(key(row))
        rows.append(row)
    return predict(pd.DataFrame(rows)), {"starts_tried": tried, "skipped_no_target": skipped}


def prepare():
    if (OUT / "designs_and_predictions_before_simulation.json").exists():
        raise SystemExit("designs_and_predictions_before_simulation.json exists: the designs are fixed. Delete it only to start a new test.")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    a = group_a(rng).assign(group="A")
    pool = random_pool(rng)
    b, b_log = group_b(pool, rng)
    b = b.assign(group="B")
    c, c_log = group_c(pool, rng)
    c = c.assign(group="C")
    d = pd.concat([a, b, c], ignore_index=True)
    known, seen, keep = known_keys(), set(), []
    for i, row in d.iterrows():
        k = key(row)
        keep.append(k not in known and k not in seen)
        seen.add(k)
    print(f"dropped as duplicates of the 560 cases or of each other: {len(d) - sum(keep)}")
    d = d[keep].reset_index(drop=True)
    d["id"] = [f"{g}{i + 1:03d}" for g, i in zip(d["group"], d.groupby("group").cumcount())]
    margins = decision.decision_config()["margins_pp"]
    protocol = {
        "purpose": "independent check of accuracy, 90 % margins, decision rule and suggestions",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "pool_size": POOL,
        "group_sizes": d["group"].value_counts().sort_index().to_dict(),
        "margins_pp_fixed_from_the_test_set": margins,
        "rules": ["designs and predictions are fixed before any simulation",
                  "the results do not change the models, the margins or the rule",
                  "general coverage is estimated from group A only",
                  "groups B and C are selected on purpose and are reported separately"],
        "group_B_selection": b_log, "group_C_selection": c_log,
    }
    (OUT / "test_protocol.json").write_text(json.dumps(protocol, indent=2, default=float), encoding="utf-8")
    d = d.assign(facades=d["facades"].apply(list))
    d.to_json(OUT / "designs_and_predictions_before_simulation.json", orient="records", indent=1)
    print(json.dumps(protocol, indent=1, default=float))


# -- simulation -------------------------------------------------------------------

def simulate(row):
    out = RUNS / row["id"]
    if not (out / "metrics.json").exists():
        case = decision.make_case(row["facades"], row["window_type"], row["window_width_m"], row["window_height_m"],
                                  row["sill_height_m"], row["overhang_depth_m"], model_id=row["id"])
        case_file = RUNS / f"{row['id']}_case.json"
        case_file.write_text(json.dumps(case, indent=2), encoding="utf-8")
        proc = subprocess.run([sys.executable, str(ROOT / "2_radiance" / "annual_metrics.py"), "--case-json",
                               str(case_file), "--out", str(out)], stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, cwd=str(ROOT))
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-2000:])
    return row["id"]


def run(workers):
    d = pd.read_json(OUT / "designs_and_predictions_before_simulation.json", orient="records")
    RUNS.mkdir(parents=True, exist_ok=True)
    start, done, failed = time.perf_counter(), 0, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(simulate, r): r["id"] for r in d.to_dict("records")}
        for fut in as_completed(futures):
            done += 1
            try:
                fut.result()
                print(f"[{done}/{len(d)}] {futures[fut]}  {time.perf_counter() - start:.0f}s", flush=True)
            except Exception as err:
                failed.append(futures[fut])
                print(f"[{done}/{len(d)}] {futures[fut]} FAILED: {err}", flush=True)
    print(f"finished in {time.perf_counter() - start:.0f}s, failed: {failed}")


# -- analysis ---------------------------------------------------------------------

def clopper_pearson(k, n, level=0.95):
    a = 1 - level
    return (0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k)))


def accuracy(d, t, margin):
    err = d[f"pred_{t}"] - d[f"sim_{t}"]
    covered = int((err.abs() <= margin).sum())
    lo, hi = clopper_pearson(covered, len(d))
    return {"n": len(d), "MAE_pp": err.abs().mean(), "RMSE_pp": float(np.sqrt((err ** 2).mean())), "bias_pp": err.mean(),
            "max_abs_pp": err.abs().max(), "margin_pp": margin, "within_margin": covered,
            "coverage_percent": 100 * covered / len(d), "coverage_ci95_percent": [100 * lo, 100 * hi],
            "R2": 1 - (err ** 2).sum() / ((d[f"sim_{t}"] - d[f"sim_{t}"].mean()) ** 2).sum()}


def figure(a, margins):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    for ax, t, label in zip(axes, common.TARGETS, (common.TARGET_LABELS[x] for x in common.TARGETS)):
        lim = [min(a[f"sim_{t}"].min(), a[f"pred_{t}"].min()) - 3, max(a[f"sim_{t}"].max(), a[f"pred_{t}"].max()) + 3]
        m = margins[t]
        ax.fill_between(lim, [lim[0] - m, lim[1] - m], [lim[0] + m, lim[1] + m], color=GRID, linewidth=0, zorder=0)
        ax.plot(lim, lim, color=INK2, linewidth=1.0, zorder=1)
        ax.scatter(a[f"sim_{t}"], a[f"pred_{t}"], s=22, color=BLUE, edgecolors="white", linewidths=0.5, zorder=3)
        ax.set(xlim=lim, ylim=lim, xlabel=f"Radiance {label} [%]", ylabel=f"MLP ensemble {label} [%]")
        ax.set_title(f"{label}: band shows ±{m:.1f} pp", loc="left")
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "fig_independent_validation_groupA.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyse():
    d = pd.read_json(OUT / "designs_and_predictions_before_simulation.json", orient="records")
    margins = decision.decision_config()["margins_pp"]
    sims = []
    for i in d["id"]:
        path = RUNS / i / "metrics.json"
        if path.exists():
            m = json.loads(path.read_text(encoding="utf-8"))
            sims.append({"id": i, f"sim_{SDA}": m[SDA], f"sim_{ASE}": m[ASE], "seconds": m.get("total_model_s")})
    if not sims:   # the Radiance run folders are not stored in the repository: use the saved Radiance results
        saved = pd.read_csv(OUT / "predicted_vs_radiance_147_designs.csv")
        sims = saved[["id", f"sim_{SDA}", f"sim_{ASE}", "seconds"]].to_dict("records")
    d = d.merge(pd.DataFrame(sims), on="id")
    for t in common.TARGETS:
        d[f"abs_err_{t}"] = (d[f"pred_{t}"] - d[f"sim_{t}"]).abs()
        d[f"level_{t}"] = d[f"spread_{t}"].apply(lambda s, t=t: decision.spread_level(t, s))
        d[f"in_range_{t}"] = False
        for thr in decision.LEED_THRESHOLDS[t]:
            d[f"in_range_{t}"] |= (d[f"pred_{t}"] - thr).abs() < margins[t]
    d["rule"] = np.where(d[f"in_range_{SDA}"] | d[f"in_range_{ASE}"], "radiance", "mlp")
    d["leed_pred"] = [decision.leed_level(s, a) for s, a in zip(d[f"pred_{SDA}"], d[f"pred_{ASE}"])]
    d["leed_sim"] = [decision.leed_level(s, a) for s, a in zip(d[f"sim_{SDA}"], d[f"sim_{ASE}"])]
    d["leed_wrong"] = d["leed_pred"] != d["leed_sim"]
    d.drop(columns=["facades"]).assign(facades=d["facades"].apply("+".join)).to_csv(
        OUT / "predicted_vs_radiance_147_designs.csv", index=False, float_format="%.3f")

    a, b, c = (d[d["group"] == g] for g in "ABC")
    out = {"n_simulated": len(d), "groups": d["group"].value_counts().sort_index().to_dict(),
           "seconds_per_simulation": {"median": d["seconds"].median(), "max": d["seconds"].max()}}

    # A: general accuracy, coverage and rule
    out["A"] = {"accuracy": {SHORT[t]: accuracy(a, t, margins[t]) for t in common.TARGETS},
                "rule": {"sent_to_radiance": int((a["rule"] == "radiance").sum()),
                         "accepted": int((a["rule"] == "mlp").sum()),
                         "leed_wrong_all": int(a["leed_wrong"].sum()),
                         "leed_wrong_accepted": int((a["leed_wrong"] & (a["rule"] == "mlp")).sum()),
                         "leed_wrong_caught": int((a["leed_wrong"] & (a["rule"] == "radiance")).sum())},
                "spread": {}}
    for t in common.TARGETS:
        rho = spearmanr(a[f"spread_{t}"], a[f"abs_err_{t}"]).statistic
        groups = a.groupby(f"level_{t}")[f"abs_err_{t}"].agg(["mean", "count"]).reindex(["low", "medium", "high"])
        out["A"]["spread"][SHORT[t]] = {"spearman": float(rho), "MAE_by_level": groups["mean"].round(3).to_dict(),
                                        "n_by_level": groups["count"].to_dict(),
                                        "within_1sd_percent": float(100 * (a[f"abs_err_{t}"] <= a[f"spread_{t}"]).mean())}

    # B: side of each threshold, per cell
    rows = []
    for (t, thr, cell), g in b.groupby(["threshold_metric", "threshold_value", "cell"]):
        wrong = ((g[f"pred_{t}"] >= thr) != (g[f"sim_{t}"] >= thr)) if t == SDA else ((g[f"pred_{t}"] > thr) != (g[f"sim_{t}"] > thr))
        rows.append({"threshold": f"{SHORT[t]} {thr:g}%", "cell": cell, "n": len(g), "wrong_side": int(wrong.sum()),
                     "MAE_pp": g[f"abs_err_{t}"].mean(), "leed_wrong": int(g["leed_wrong"].sum()),
                     "other_metric_decisive": bool(g["other_metric_decisive"].all())})
    out["B"] = {"cells": rows,
                "accepted_by_rule": int((b["rule"] == "mlp").sum()),
                "accepted_leed_wrong": int((b["leed_wrong"] & (b["rule"] == "mlp")).sum()),
                "sent_to_radiance": int((b["rule"] == "radiance").sum()),
                "caught_leed_wrong": int((b["leed_wrong"] & (b["rule"] == "radiance")).sum()),
                "accuracy": {SHORT[t]: accuracy(b, t, margins[t]) for t in common.TARGETS}}

    # C: selection bias of the suggestions, compared with the general bias of group A
    target_met = [decision.leed_level(s, x).startswith(f"{int(p)} points") or (int(p) == 2 and decision.leed_level(s, x).startswith("3"))
                  for s, x, p in zip(c[f"sim_{SDA}"], c[f"sim_{ASE}"], c["target_points"])]
    out["C"] = {"accuracy": {SHORT[t]: accuracy(c, t, margins[t]) for t in common.TARGETS},
                "target_reached_in_radiance": int(sum(target_met)), "n": len(c),
                "target_points": c["target_points"].value_counts().to_dict(),
                "favourable_error_share_percent": {"sDA_overestimated": float(100 * (c[f"pred_{SDA}"] > c[f"sim_{SDA}"]).mean()),
                                                   "ASE_underestimated": float(100 * (c[f"pred_{ASE}"] < c[f"sim_{ASE}"]).mean())},
                "spread_levels": {SHORT[t]: c[f"level_{t}"].value_counts().to_dict() for t in common.TARGETS},
                "bias_difference_C_minus_A_pp": {SHORT[t]: float((c[f"pred_{t}"] - c[f"sim_{t}"]).mean() - (a[f"pred_{t}"] - a[f"sim_{t}"]).mean())
                                                  for t in common.TARGETS}}
    (OUT / "accuracy_and_coverage_by_group.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    figure(a, margins)
    print(json.dumps(out, indent=1, default=float))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--analyse", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.run:
        run(args.workers)
    if args.analyse:
        analyse()


if __name__ == "__main__":
    main()
