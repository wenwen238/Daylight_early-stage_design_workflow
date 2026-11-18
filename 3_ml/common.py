"""Shared data handling, tuning and reporting for the sDA/ASE surrogate models.

train_xgboost.py, train_mlp.py and the baseline train_linear.py call
run_experiment() with their own estimator and hyperparameter search space, so all
models are trained on the same data, the same stratified 80/20 split and the same
cross-validation folds. The final fitted models are saved with joblib.

Protocol
--------
1. 80/20 split of the 560 cases, stratified by design stratum (8 train + 2 test
   cases per stratum). The test set is used once, after tuning.
2. Hyperparameters are tuned by randomised search with 5-fold cross-validation
   on the 448 training cases; the folds are stratified by design stratum.
   The configuration with the lowest mean cross-validated RMSE is selected.
3. The selected configuration is refitted on all training cases and evaluated
   on the test set. Predictions are clipped to 0-100 % in the cross-validation
   scores and in the final evaluation alike.
"""

import argparse
import json
import platform
import shutil
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

DEFAULT_CASES = config.CASES_CSV
DEFAULT_RESULTS = config.RADIANCE_SUMMARY
OUT_ROOT = config.ML_TRAINING

TARGETS = ["sDA_300_50_percent", "ASE_1000_250_percent"]
TARGET_LABELS = {"sDA_300_50_percent": "sDA300/50%", "ASE_1000_250_percent": "ASE1000,250h"}
TARGET_RANGE = (0.0, 100.0)          # both metrics are area percentages
# The windowed façades are encoded as one 0/1 flag per orientation, so that e.g.
# "has a south window" is shared by every pattern that contains a south window.
FACADES = ["north", "east", "south", "west"]
FACADE_FLAGS = [f"has_{f}_window" for f in FACADES]
CATEGORICAL = ["window_type", "overhang_present"]
NUMERICAL = ["sill_height_m", "window_width_m", "window_height_m", "overhang_depth_m"]
# "reduced" drops the two categorical features, which are fully determined by the
# numerical ones (window_type by sill_height_m, overhang_present by overhang_depth_m > 0).
FEATURE_SETS = {
    "full": {"categorical": CATEGORICAL, "numerical": NUMERICAL},
    "reduced": {"categorical": [], "numerical": NUMERICAL},
}
RANDOM_STATE = 20260924
TEST_SIZE = 0.20
CV_FOLDS = 5
N_JOBS = 8                            # leave part of the machine free


def parse_args(description, default_iter):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--n-iter", type=int, default=default_iter,
                        help="randomised-search configurations per target")
    parser.add_argument("--features", choices=list(FEATURE_SETS), default="reduced")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def feature_columns(feature_set):
    spec = FEATURE_SETS[feature_set]
    return FACADE_FLAGS + spec["categorical"] + spec["numerical"]


# -- data ---------------------------------------------------------------

def load_dataset(cases_path, results_path):
    cases = pd.read_csv(cases_path, dtype={"model_id": str})
    results = pd.read_csv(results_path, dtype={"model_id": str})
    required_cases = ["model_id", "stratum_id", "active_facades"] + CATEGORICAL + NUMERICAL
    required_results = ["model_id"] + TARGETS
    missing = [c for c in required_cases if c not in cases] + [c for c in required_results if c not in results]
    if missing:
        raise ValueError(f"Required column(s) not found: {sorted(set(missing))}")

    data = cases[required_cases].merge(results[required_results], on="model_id", how="inner")
    if len(data) != len(cases):
        missing_models = sorted(set(cases["model_id"]) - set(data["model_id"]))
        raise ValueError(
            f"Only {len(data)} of {len(cases)} sampling cases have simulation results. "
            f"Missing examples: {missing_models[:10]}"
        )
    if len(data) != 560:
        raise ValueError(f"Expected 560 completed cases, found {len(data)}.")

    data[CATEGORICAL] = data[CATEGORICAL].astype(str)
    data[NUMERICAL + TARGETS] = data[NUMERICAL + TARGETS].apply(pd.to_numeric, errors="raise")
    facades = data["active_facades"].str.split(";")
    unknown = set().union(*facades) - set(FACADES)
    if unknown:
        raise ValueError(f"Unknown façade name(s) in active_facades: {sorted(unknown)}")
    for facade, flag in zip(FACADES, FACADE_FLAGS):
        data[flag] = facades.apply(lambda names: int(facade in names))
    return data.sort_values("model_id").reset_index(drop=True)


def split_dataset(data):
    """Stratified by design stratum: every one of the 56 strata keeps 8 training and 2 test cases."""
    train, test = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True,
                                   stratify=data["stratum_id"])
    return train.sort_values("model_id").reset_index(drop=True), test.sort_values("model_id").reset_index(drop=True)


def cv_folds(train):
    """Five folds on the training cases, stratified by design stratum (identical for both models)."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return list(skf.split(train, train["stratum_id"]))


def one_hot_encoder():
    """Support both current and older scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(scale_numeric, feature_set):
    spec = FEATURE_SETS[feature_set]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    transformers = [("facades", "passthrough", FACADE_FLAGS),
                    ("numeric", Pipeline(numeric_steps), spec["numerical"])]
    if spec["categorical"]:
        categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                ("onehot", one_hot_encoder())])
        transformers.append(("categorical", categorical, spec["categorical"]))
    return ColumnTransformer(transformers, sparse_threshold=0)


# -- metrics ------------------------------------------------------------

def clip(pred):
    return np.clip(pred, *TARGET_RANGE)


def rmse(y, pred):
    return float(np.sqrt(mean_squared_error(y, pred)))


def clipped_rmse(y, pred):
    return rmse(y, clip(pred))


def clipped_mae(y, pred):
    return mean_absolute_error(y, clip(pred))


def clipped_r2(y, pred):
    return r2_score(y, clip(pred))


SCORERS = {
    "rmse": make_scorer(clipped_rmse, greater_is_better=False),
    "mae": make_scorer(clipped_mae, greater_is_better=False),
    "r2": make_scorer(clipped_r2),
}


def scores(y, pred):
    return {"R2": r2_score(y, pred), "MAE_percent_points": mean_absolute_error(y, pred),
            "RMSE_percent_points": rmse(y, pred)}


# -- experiment ---------------------------------------------------------

def make_plot(predictions, model_name, color, out_path):
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(10, 4.5))
    for ax, target in zip(np.atleast_1d(axes), TARGETS):
        actual = predictions[f"actual_{target}"]
        predicted = predictions[f"predicted_{target}"]
        lo = min(actual.min(), predicted.min())
        hi = max(actual.max(), predicted.max())
        pad = max((hi - lo) * 0.05, 1)
        ax.scatter(actual, predicted, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#6b7280", linewidth=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{TARGET_LABELS[target]} (test set)")
        ax.set_xlabel("Radiance result [%]")
        ax.set_ylabel(f"{model_name} prediction [%]")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def package_versions():
    versions = {"python": platform.python_version()}
    for pkg in ("scikit-learn", "xgboost", "numpy", "pandas", "joblib"):
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return versions


def run_experiment(args, model_name, out_name, make_estimator, search_space, color, importance_fn=None,
                   tuned=True):
    """Tune, refit and evaluate one model type for both targets; write all outputs.

    With tuned=False the search space holds a single fixed configuration; the
    cross-validation then only reports how stable the model is across folds.
    """
    start = time.perf_counter()
    out_dir = OUT_ROOT / args.features / out_name
    features = feature_columns(args.features)
    data = load_dataset(args.cases, args.results)
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_dir} already exists. Use --overwrite to replace it.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    train, test = split_dataset(data)
    folds = cv_folds(train)
    train[["model_id"]].to_csv(out_dir / "train_model_ids.csv", index=False)
    test[["model_id"]].to_csv(out_dir / "test_model_ids.csv", index=False)

    predictions = test[["model_id"]].copy()
    metrics, cv_summary, best_params, importance_rows = [], [], {}, []
    for target in TARGETS:
        t0 = time.perf_counter()
        search = RandomizedSearchCV(
            make_estimator(args.features), search_space, n_iter=args.n_iter, scoring=SCORERS, refit="rmse",
            cv=folds, n_jobs=N_JOBS, random_state=RANDOM_STATE, return_train_score=False,
        )
        search.fit(train[features], train[target])
        tune_s = time.perf_counter() - t0

        cv = pd.DataFrame(search.cv_results_)
        param_cols = [c for c in cv if c.startswith("param_")]
        keep = ["rank_test_rmse"] + [f"{s}_test_{m}" for m in SCORERS for s in ("mean", "std")] + param_cols
        table = cv[keep].sort_values("rank_test_rmse")
        table = table.rename(columns={c: c.split("__")[-1] for c in param_cols})
        for m in ("rmse", "mae"):              # scorers are negated for maximisation; report positive errors
            table[f"mean_test_{m}"] = -table[f"mean_test_{m}"]
        table.to_csv(out_dir / f"cv_results_{target}.csv", index=False, float_format="%.6f")
        best = cv.loc[search.best_index_]
        cv_summary.append({
            "target": target, "n_configurations": args.n_iter, "cv_folds": CV_FOLDS,
            "cv_R2_mean": best["mean_test_r2"], "cv_R2_std": best["std_test_r2"],
            "cv_MAE_mean": -best["mean_test_mae"], "cv_MAE_std": best["std_test_mae"],
            "cv_RMSE_mean": -best["mean_test_rmse"], "cv_RMSE_std": best["std_test_rmse"],
            "tuning_time_s": tune_s,
        })
        best_params[target] = {k: (v if not isinstance(v, (np.floating, np.integer)) else v.item())
                               for k, v in search.best_params_.items()}

        model = search.best_estimator_                 # refitted on all training cases
        joblib.dump(model, out_dir / f"model_{target}.joblib")
        for split_name, split in (("train", train), ("test", test)):
            pred = clip(model.predict(split[features]))
            metrics.append({"target": target, "split": split_name, **scores(split[target], pred),
                            "n_cases": len(split)})
            if split_name == "test":
                predictions[f"actual_{target}"] = split[target].to_numpy()
                predictions[f"predicted_{target}"] = pred
        if importance_fn:
            importance_rows.extend({"target": target, **row} for row in importance_fn(model))
        print(f"{model_name} {target}: tuned in {tune_s:.0f}s, best CV RMSE "
              f"{-best['mean_test_rmse']:.2f} ± {best['std_test_rmse']:.2f}", flush=True)

    pd.DataFrame(metrics).to_csv(out_dir / "metrics.csv", index=False, float_format="%.6f")
    pd.DataFrame(cv_summary).to_csv(out_dir / "cv_summary.csv", index=False, float_format="%.6f")
    predictions.to_csv(out_dir / "test_predictions.csv", index=False, float_format="%.6f")
    (out_dir / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str), encoding="utf-8")
    if importance_rows:
        pd.DataFrame(importance_rows).sort_values(["target", "importance"], ascending=[True, False]).to_csv(
            out_dir / "feature_importance.csv", index=False, float_format="%.6f")
    make_plot(predictions, model_name, color, out_dir / "test_predictions.png")
    (out_dir / "run_settings.json").write_text(json.dumps({
        "model": model_name, "random_state": RANDOM_STATE, "n_cases": len(data),
        "split": f"{int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} stratified by stratum_id",
        "tuning": (f"RandomizedSearchCV, {args.n_iter} configurations, {CV_FOLDS}-fold CV on the training "
                   f"cases stratified by stratum_id, selection by mean CV RMSE") if tuned else
                  f"no hyperparameters tuned; {CV_FOLDS}-fold CV on the training cases for evaluation only",
        "prediction_clip": TARGET_RANGE, "feature_set": args.features, "features": features, "targets": TARGETS,
        "search_space": {k: str(v) for k, v in search_space.items()},
        "saved_models": [f"model_{t}.joblib" for t in TARGETS],
        "package_versions": package_versions(),
        "elapsed_s": time.perf_counter() - start,
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"Saved results to: {out_dir}  ({time.perf_counter() - start:.0f}s)")
