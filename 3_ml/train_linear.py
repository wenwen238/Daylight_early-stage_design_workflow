#!/usr/bin/env python3
"""Linear-regression baseline for annual sDA and ASE.

Ordinary least squares on the same features, split and cross-validation folds as
train_xgboost.py and train_mlp.py (shared through common.py). It has no
hyperparameters to tune; the cross-validation only reports its stability. The
baseline shows how much of the variance is explained without non-linear effects
or interactions.
"""

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import common


# A single fixed configuration: RandomizedSearchCV then only runs the cross-validation.
SEARCH_SPACE = {"regressor__fit_intercept": [True]}


def make_estimator(feature_set):
    return Pipeline([
        ("preprocess", common.build_preprocessor(scale_numeric=True, feature_set=feature_set)),
        ("regressor", LinearRegression()),
    ])


def main():
    args = common.parse_args(__doc__, default_iter=1)
    common.run_experiment(args, "Linear regression", "linear", make_estimator, SEARCH_SPACE, "#6b7280",
                          tuned=False)


if __name__ == "__main__":
    main()
