#!/usr/bin/env python3
"""Extension 5: how many Radiance simulations are needed to train the surrogate? (learning curve)

The 112 test cases of the common split stay fixed. The models are trained on stratified
subsets of the 448 training cases with 1, 2, 4, 6 and 8 cases from each of the 56 design
strata (56, 112, 224, 336 and 448 cases), so every subset covers all strata. Each size is
repeated with five seeds: the seed selects the subset and initialises the model. With 8
cases per stratum the subset is always the full training set, so only the model seed changes.

Models: MLP, XGBoost and linear regression with the hyperparameters selected on the full
training set (reduced/<model>/best_params.json); they are not re-tuned for each size.
Predictions are clipped to 0-100 %, as everywhere else.

Outputs (thesis_data/results/extensions/learning_curve/): every run, a summary per size and one figure
(thesis Fig. 4.17, drawn by thesis_data/figure_scripts/fig_learning_curve.py).
"""

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
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "3_ml"))
import config  # noqa: E402
import common  # noqa: E402
import train_linear  # noqa: E402
import train_mlp  # noqa: E402
import train_xgboost  # noqa: E402

OUT = config.RESULTS / "extensions" / "learning_curve"
PER_STRATUM = [1, 2, 4, 6, 8]
SEEDS = [1, 2, 3, 4, 5]
MODELS = {   # name: (module, best_params folder, random_state parameter, colour)
    "MLP": (train_mlp, "mlp", "regressor__regressor__random_state", "#1baf7a"),
    "XGBoost": (train_xgboost, "xgboost", "regressor__random_state", "#eb6834"),
    "Linear regression": (train_linear, "linear", None, "#2a78d6"),
}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 9.5, "axes.titlecolor": INK, "axes.linewidth": 0.8,
})


def best_params(folder, target):
    p = json.loads((config.ML_TRAINING / "reduced" / folder / "best_params.json").read_text(encoding="utf-8"))[target]
    return {k: tuple(v) if isinstance(v, list) else v for k, v in p.items()}


def subset(train, k, seed):
    """k training cases from every design stratum."""
    rng = np.random.default_rng(seed)
    idx = [rng.choice(g.index.to_numpy(), size=k, replace=False) for _, g in train.groupby("stratum_id")]
    return train.loc[np.concatenate(idx)]


def run_one(name, target, k, seed, train, test, feats):
    module, folder, rs, _ = MODELS[name]
    model = module.make_estimator("reduced").set_params(**best_params(folder, target))
    if rs:
        model.set_params(**{rs: seed})
    sub = subset(train, k, seed)
    t0 = time.perf_counter()
    model.fit(sub[feats], sub[target])
    pred = common.clip(model.predict(test[feats]))
    y = test[target].to_numpy()
    return {"model": name, "target": target, "per_stratum": k, "n_train": len(sub), "seed": seed,
            "test_RMSE_pp": common.rmse(y, pred), "test_MAE_pp": mean_absolute_error(y, pred),
            "test_R2": r2_score(y, pred), "fit_s": time.perf_counter() - t0}


def figure(summary):
    """Draw thesis Fig. 4.17 with thesis_data/figure_scripts/fig_learning_curve.py (linear axis, 0-16 pp)."""
    sys.path.insert(0, str(ROOT / "thesis_data" / "figure_scripts"))
    import fig_learning_curve  # noqa: E402
    fig_learning_curve.draw(summary, OUT / "fig_learning_curve.png")


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    data = common.load_dataset(common.DEFAULT_CASES, common.DEFAULT_RESULTS)
    train, test = common.split_dataset(data)                    # same 448 / 112 split as the thesis
    feats = common.feature_columns("reduced")
    jobs = [(name, t, k, s) for name in MODELS for t in common.TARGETS for k in PER_STRATUM for s in SEEDS]
    rows = Parallel(n_jobs=common.N_JOBS)(delayed(run_one)(*j, train, test, feats) for j in jobs)
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "learning_curve_runs.csv", index=False, float_format="%.5f")
    summary = (runs.groupby(["model", "target", "per_stratum", "n_train"])
               .agg(RMSE_mean=("test_RMSE_pp", "mean"), RMSE_sd=("test_RMSE_pp", "std"),
                    RMSE_min=("test_RMSE_pp", "min"), RMSE_max=("test_RMSE_pp", "max"),
                    MAE_mean=("test_MAE_pp", "mean"), R2_mean=("test_R2", "mean"))
               .reset_index())
    summary.to_csv(OUT / "learning_curve_summary.csv", index=False, float_format="%.4f")
    figure(summary)
    print(summary.pivot_table(index="n_train", columns=["target", "model"], values="RMSE_mean").round(2).to_string())
    print(f"Saved to {OUT}  ({time.perf_counter() - start:.0f}s)")


if __name__ == "__main__":
    main()
