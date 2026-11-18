#!/usr/bin/env python3
"""Tune and train separate XGBoost regressors for annual sDA and ASE.

Data, split, cross-validation folds and evaluation are shared with train_mlp.py
through common.py, so the two models are compared under the same protocol.
"""

from scipy.stats import loguniform, uniform
from sklearn.pipeline import Pipeline

import common

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise SystemExit("XGBoost is not installed in this Python environment (see README).") from exc


SEARCH_SPACE = {
    "regressor__n_estimators": [200, 400, 800, 1500, 3000, 5000],
    "regressor__max_depth": [2, 3, 4, 5, 6, 8, 10, 12],
    "regressor__learning_rate": loguniform(0.01, 0.3),
    "regressor__subsample": uniform(0.6, 0.4),
    "regressor__colsample_bytree": uniform(0.6, 0.4),
    "regressor__min_child_weight": [1, 2, 5, 10, 20, 40],
    "regressor__reg_lambda": loguniform(1e-3, 100),
    "regressor__reg_alpha": [0.0, 0.01, 0.1, 1.0],
}


def make_estimator(feature_set):
    return Pipeline([
        ("preprocess", common.build_preprocessor(scale_numeric=False, feature_set=feature_set)),
        ("regressor", XGBRegressor(objective="reg:squarederror", n_jobs=1,
                                   random_state=common.RANDOM_STATE)),
    ])


def importance(model):
    names = model.named_steps["preprocess"].get_feature_names_out()
    values = model.named_steps["regressor"].feature_importances_
    return [{"feature": n, "importance": float(v)} for n, v in zip(names, values)]


def main():
    args = common.parse_args(__doc__, default_iter=60)
    common.run_experiment(args, "XGBoost", "xgboost", make_estimator, SEARCH_SPACE, "#2a6f97",
                          importance_fn=importance)


if __name__ == "__main__":
    main()
