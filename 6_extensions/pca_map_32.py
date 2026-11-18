#!/usr/bin/env python3
"""Check: PCA + MLP on the full 32 x 32 Radiance maps, and sDA / ASE computed from the predicted map.

Same method as pca_map_prediction.py (448 / 112 split, k by five-fold CV on the training cases,
five-seed MLP ensemble with the scalar-model hyperparameters), but on the 1024 sensors of the
Radiance grid itself, so the metric from the map uses the same definition as the thesis sDA / ASE
and there is no grid error.

Compared on the same 112 test cases with the 5-MLP ensemble that predicts sDA / ASE directly
(3_ml/train_mlp_ensemble.py), thesis Table 4.16.

    python 6_extensions/pca_map_32.py --metric da300
    python 6_extensions/pca_map_32.py --metric ase

Outputs: thesis_data/results/extensions/pca_map_32/<metric>/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "3_ml"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import common  # noqa: E402
import pca_map_prediction as pmp  # noqa: E402


def load_maps32(data):
    maps = np.array([np.load(config.RADIANCE_560 / m / pmp.CFG["file"]).astype(float).ravel()
                     for m in data.model_id])
    return maps * pmp.CFG["scale"]


def errors(pred, true):
    e = pred - true
    return {"MAE_pp": float(np.abs(e).mean()), "RMSE_pp": float(np.sqrt((e ** 2).mean())),
            "max_abs_pp": float(np.abs(e).max()), "bias_pp": float(e.mean()),
            "share_within_1pp": float((np.abs(e) <= 1.0).mean()),
            "share_within_2pp": float((np.abs(e) <= 2.0).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=list(pmp.METRICS), default="da300")
    metric = parser.parse_args().metric
    pmp.CFG = pmp.METRICS[metric]
    cfg, u = pmp.CFG, pmp.CFG["unit"]
    out = config.RESULTS / "extensions" / "pca_map_32" / metric
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    data = common.load_dataset(common.DEFAULT_CASES, common.DEFAULT_RESULTS)
    feats = common.feature_columns("reduced")
    maps = pd.DataFrame(load_maps32(data), index=data.model_id)
    train, test = common.split_dataset(data)
    Ytr, Yte = maps.loc[train.model_id].to_numpy(), maps.loc[test.model_id].to_numpy()
    Xtr, Xte = train[feats], test[feats]
    official = test[cfg["target"]].to_numpy()
    check = np.abs(pmp.metric_from_map(Yte) - official).max()
    print(f"metric from the simulated 32 x 32 maps vs thesis values: max difference {check:.4f} pp")

    cum = np.cumsum(PCA().fit(Ytr).explained_variance_ratio_)
    k_var = int(np.argmax(cum >= pmp.VAR_TARGET) + 1)
    folds = common.cv_folds(train)
    k_all = sorted(set(pmp.K_LIST + [k_var]))
    jobs = [(k, f) for k in k_all for f in range(len(folds))]
    cv_rmse = Parallel(n_jobs=common.N_JOBS)(delayed(pmp.cv_one)(k, folds[f], Xtr, Ytr, cfg) for k, f in jobs)
    cv = pd.DataFrame([{"k": k, "fold": f, "rmse": r} for (k, f), r in zip(jobs, cv_rmse)])
    cv = cv.groupby("k").rmse.agg(["mean", "std"]).reset_index()
    cv.columns = ["k", f"cv_RMSE_{u}", f"cv_RMSE_sd_{u}"]
    cv.to_csv(out / "pca_k_cv.csv", index=False, float_format="%.4f")
    k_sel = int(cv.loc[cv[f"cv_RMSE_{u}"].idxmin(), "k"])
    print(cv.round(2).to_string(index=False), f"\nk selected by CV: {k_sel} (99.5 % variance would give {k_var})")

    pca = PCA(n_components=k_sel).fit(Ytr)
    pred = pmp.clip(pca.inverse_transform(pmp.ensemble_predict(Xtr, pca.transform(Ytr), Xte)))
    m_pred = pmp.metric_from_map(pred)

    name = "sDA_300_50_percent" if cfg["at_least"] else "ASE_1000_250_percent"
    ens = pd.read_csv(config.ML_TRAINING / "reduced" / "mlp_ensemble" / "ensemble_test_predictions.csv",
                      index_col="model_id").loc[test.model_id]

    comparison = pd.DataFrame([
        {"method": "PCA + MLP, 32 x 32 map -> metric", **errors(m_pred, official)},
        {"method": "5-MLP ensemble, metric directly", **errors(ens[f"mean_{name}"].to_numpy(), official)},
    ])
    comparison.to_csv(out / "metric_comparison.csv", index=False, float_format="%.4f")
    err = pred - Yte
    summary = {"metric": metric, "grid": "32 x 32 Radiance sensors (0.1875 m, cell centres)",
               "k_selected_by_cv": k_sel, "k_for_99_5_percent_variance": k_var,
               "explained_variance_k": float(cum[k_sel - 1]),
               f"point_RMSE_{u}": float(np.sqrt((err ** 2).mean())), f"point_MAE_{u}": float(np.abs(err).mean()),
               "metric_check_max_diff_pp": float(check), "n_train": len(train), "n_test": len(test),
               "seeds": pmp.SEEDS, "elapsed_s": time.perf_counter() - start}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame({"model_id": test.model_id.to_numpy(), "official": official, "from_pca_map_32": m_pred,
                  "ensemble_direct": ens[f"mean_{name}"].to_numpy()}
                 ).to_csv(out / "test_metric_predictions.csv", index=False, float_format="%.4f")
    print(json.dumps(summary, indent=1))
    print(comparison.round(3).to_string(index=False))
    print(f"Saved to {out}  ({time.perf_counter() - start:.0f}s)")


if __name__ == "__main__":
    main()
