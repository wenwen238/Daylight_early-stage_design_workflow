#!/usr/bin/env python3
"""Tune and train separate scikit-learn MLP regressors for annual sDA and ASE.

Data, split, cross-validation folds and evaluation are shared with
train_xgboost.py through common.py, so the two models are compared under the
same protocol. Numerical inputs and the target are standardised.
"""

from scipy.stats import loguniform
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import common


P = "regressor__regressor__"          # TransformedTargetRegressor -> Pipeline -> MLPRegressor
SEARCH_SPACE = {
    P + "hidden_layer_sizes": [(32,), (64,), (128,), (32, 32), (64, 32), (64, 64), (128, 64), (128, 64, 32)],
    P + "activation": ["relu", "tanh"],
    P + "alpha": loguniform(1e-5, 1e-1),
    P + "learning_rate_init": loguniform(1e-4, 1e-2),
    P + "batch_size": [16, 32, 64],
}


def make_estimator(feature_set):
    network = Pipeline([
        ("preprocess", common.build_preprocessor(scale_numeric=True, feature_set=feature_set)),
        ("regressor", MLPRegressor(
            solver="adam", max_iter=3000, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=100, random_state=common.RANDOM_STATE,
        )),
    ])
    return TransformedTargetRegressor(regressor=network, transformer=StandardScaler())


def main():
    args = common.parse_args(__doc__, default_iter=60)
    common.run_experiment(args, "MLP", "mlp", make_estimator, SEARCH_SPACE, "#8a5aab")


if __name__ == "__main__":
    main()
