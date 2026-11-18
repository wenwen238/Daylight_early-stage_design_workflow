#!/usr/bin/env python3
"""PCA + five-seed MLP ensembles that predict the whole DA300 map and ASE hour map (10 x 10 points).

Maps
  The Radiance results of the 560 cases contain a 32 x 32 DA300 map (share of occupied hours
  with at least 300 lx) and a 32 x 32 map of occupied hours with direct sun above 1000 lx.
  Both are resampled by bilinear interpolation to the 10 x 10 cell centres of a 0.6 m grid
  (x, y = 0.3, 0.9, ..., 5.7 m from the south-west corner; rows from south to north,
  columns from west to east), which cover the whole floor. All points lie inside the 32 x 32 grid.

Model
  1. PCA of the 10 x 10 maps of the 448 training cases of the common split: each map = mean map
     + k patterns x scores. k = 10 (DA300) and 15 (ASE), the values selected by five-fold
     cross-validation on the training cases in 6_extensions/pca_map_prediction.py.
  2. An MLP predicts the k scores from the eight design inputs. As for the sDA/ASE ensemble
     (train_mlp_ensemble.py), five networks with the seeds 20260924-20260928 are trained and their
     predicted scores are averaged. The hyperparameters are those selected for the matching scalar
     model (sDA for the DA300 map, ASE for the ASE map; reduced/mlp/best_params.json); they are not
     re-tuned for the maps.
  3. The predicted map is the PCA reconstruction of the mean scores, limited to the valid range.

The maps are additional, advisory output: the decision rule uses only sDA and ASE of the
32 x 32 grid, which remain the values reported by the workflow.

Outputs (thesis_data/results/ml_training/reduced/map_ensemble/): per map the fitted PCA and the five MLPs,
test predictions, metrics and run settings.
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

import common
import train_mlp

SEEDS = [20260924, 20260925, 20260926, 20260927, 20260928]   # as train_mlp_ensemble.py
OUT = common.OUT_ROOT / "reduced" / "map_ensemble"
ROOM = 6.0
N32, NG = 32, 10
X32 = (np.arange(N32) + 0.5) * ROOM / N32                    # Radiance sensor positions [m]
XG = (np.arange(NG) + 0.5) * ROOM / NG                       # map points [m]: 0.3 ... 5.7
MAPS = {   # name: (file in radiance_560/<model>/, scale, valid range, unit, scalar model whose settings are reused, k)
    "da300": ("da300_map.npy", 100.0, (0.0, 100.0), "%", "sDA_300_50_percent", 10),
    "ase_hours": ("ase_hours_map.npy", 1.0, (0.0, 3650.0), "h", "ASE_1000_250_percent", 15),
}


def to_grid(map32):
    """Bilinear interpolation of a 32 x 32 map to the 10 x 10 points."""
    f = RegularGridInterpolator((X32, X32), map32, method="linear")
    yy, xx = np.meshgrid(XG, XG, indexing="ij")
    return f(np.stack([yy.ravel(), xx.ravel()], axis=1))


def load_maps(model_ids, file, scale):
    return np.array([to_grid(np.load(common.DEFAULT_RESULTS.parent / m / file).astype(float))
                     for m in model_ids]) * scale


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    data = common.load_dataset(common.DEFAULT_CASES, common.DEFAULT_RESULTS)
    train, test = common.split_dataset(data)                    # same 448 / 112 split
    feats = common.feature_columns("reduced")
    best = json.load(open(common.OUT_ROOT / "reduced" / "mlp" / "best_params.json", encoding="utf-8"))

    metrics, predictions = [], {}
    for name, (file, scale, valid, unit, target, k) in MAPS.items():
        Ytr, Yte = load_maps(train.model_id, file, scale), load_maps(test.model_id, file, scale)
        pca = PCA(n_components=k).fit(Ytr)
        joblib.dump(pca, OUT / f"pca_{name}.joblib")
        Str = pca.transform(Ytr)
        params = {p: tuple(v) if isinstance(v, list) else v for p, v in best[target].items()}
        scores, member_maps = [], []
        for seed in SEEDS:
            m = train_mlp.make_estimator("reduced").set_params(**params, regressor__regressor__random_state=seed)
            m.fit(train[feats], Str)                            # k outputs, all 448 training cases
            joblib.dump(m, OUT / f"model_{name}_scores_seed{seed}.joblib")
            s = m.predict(test[feats]).reshape(len(test), -1)
            scores.append(s)
            member_maps.append(np.clip(pca.inverse_transform(s), *valid))
        mean = np.clip(pca.inverse_transform(np.mean(scores, axis=0)), *valid)
        err = mean - Yte
        per_map = np.sqrt((err ** 2).mean(axis=1))
        metrics.append({"map": name, "unit": unit, "k": k,
                        "explained_variance": float(pca.explained_variance_ratio_.sum()),
                        "point_RMSE": float(np.sqrt((err ** 2).mean())),
                        "point_MAE": float(np.abs(err).mean()), "R2_all_points": float(r2_score(Yte.ravel(), mean.ravel())),
                        "map_RMSE_median": float(np.median(per_map)), "map_RMSE_max": float(per_map.max()),
                        "member_point_RMSE_min": float(min(np.sqrt(((p - Yte) ** 2).mean()) for p in member_maps)),
                        "member_point_RMSE_max": float(max(np.sqrt(((p - Yte) ** 2).mean()) for p in member_maps))})
        predictions[name] = pd.DataFrame(mean, index=test.model_id,
                                         columns=[f"p{r}_{c}" for r in range(NG) for c in range(NG)])
        print(f"{name}: k = {k}, test RMSE per point {metrics[-1]['point_RMSE']:.2f} {unit}, "
              f"R2 {metrics[-1]['R2_all_points']:.4f}", flush=True)

    pd.DataFrame(metrics).to_csv(OUT / "map_metrics.csv", index=False, float_format="%.4f")
    for name, df in predictions.items():
        df.to_csv(OUT / f"test_predicted_{name}_map_10x10.csv", float_format="%.3f")
    (OUT / "run_settings.json").write_text(json.dumps({
        "method": "PCA of the training maps + five-seed MLP ensemble predicting the PCA scores",
        "members": len(SEEDS), "seeds": SEEDS, "points_per_map": NG * NG,
        "grid": "10 x 10 cell centres at 0.3, 0.9, ..., 5.7 m; rows south to north, columns west to east; "
                "bilinear interpolation of the 32 x 32 Radiance maps",
        "maps": {n: {"file": v[0], "scale": v[1], "valid_range": v[2], "unit": v[3], "hyperparameters_of": v[4],
                     "pca_components": v[5]} for n, v in MAPS.items()},
        "training_cases": len(train), "test_cases": len(test), "feature_set": "reduced",
        "package_versions": common.package_versions(), "elapsed_s": time.perf_counter() - start,
    }, indent=2), encoding="utf-8")
    print(f"Saved to {OUT}  ({time.perf_counter() - start:.0f}s)")


if __name__ == "__main__":
    main()
