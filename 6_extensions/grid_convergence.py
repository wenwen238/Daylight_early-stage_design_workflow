#!/usr/bin/env python3
"""Grid convergence: how far sDA / ASE of a coarser sensor grid are from the 32 x 32 thesis values.

No new Radiance run: the stored 32 x 32 maps (cell centres, 0.1875 m) of all 560 cases are
resampled by bilinear interpolation to n x n cell-centre grids (points at (i + 0.5) * 6 / n m),
and sDA300/50% and ASE1000,250h are counted on each grid (thesis Fig. 4.6).
Interpolated values are smoother than a simulation at those points, so the coarse-grid
differences are a lower estimate.

    python 6_extensions/grid_convergence.py

Outputs: thesis_data/results/extensions/grid_convergence/grid_convergence.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "3_ml"))
import config  # noqa: E402
import common  # noqa: E402

OUT = config.RESULTS / "extensions" / "grid_convergence"
W = 6.0
X32 = (np.arange(32) + 0.5) * W / 32
SIZES = [4, 6, 8, 10, 12, 16, 20, 24, 32]
METRICS = {"sDA": ("da300_map.npy", 100.0, 50.0, True, "sDA_300_50_percent"),
           "ASE": ("ase_hours_map.npy", 1.0, 250.0, False, "ASE_1000_250_percent")}


def resample(map32, pts):
    f = RegularGridInterpolator((X32, X32), map32, method="linear")
    yy, xx = np.meshgrid(pts, pts, indexing="ij")
    return f(np.stack([yy.ravel(), xx.ravel()], axis=1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = common.load_dataset(common.DEFAULT_CASES, common.DEFAULT_RESULTS)
    grids = {f"{n} x {n}": (np.arange(n) + 0.5) * W / n for n in SIZES}
    rows = []
    for name, (file, scale, thr, at_least, target) in METRICS.items():
        maps = [np.load(config.RADIANCE_560 / m / file).astype(float) * scale for m in data.model_id]
        ref = data[target].to_numpy()
        for g, pts in grids.items():
            vals = np.array([resample(m, pts) for m in maps])
            hit = vals >= thr if at_least else vals > thr
            e = hit.mean(axis=1) * 100 - ref
            rows.append({"metric": name, "grid": g, "points": len(pts) ** 2,
                         "spacing_m": W / len(pts),
                         "step_pp": 100 / len(pts) ** 2, "MAE_pp": np.abs(e).mean(),
                         "RMSE_pp": np.sqrt((e ** 2).mean()), "max_abs_pp": np.abs(e).max(),
                         "bias_pp": e.mean(), "share_within_1pp": (np.abs(e) <= 1).mean()})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "grid_convergence.csv", index=False, float_format="%.4f")
    print(table.round(3).to_string(index=False))
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
