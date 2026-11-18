# ML model training and comparison

Three model types are tuned, trained and evaluated separately for the annual
simulation outputs `sDA_300_50_percent` and `ASE_1000_250_percent`:

- `train_xgboost.py` — gradient-boosted trees (XGBoost);
- `train_mlp.py` — multilayer perceptron (scikit-learn `MLPRegressor`);
- `train_linear.py` — ordinary least-squares baseline without hyperparameters.

`common.py` holds everything the models share (data, features, split,
cross-validation folds, scoring and outputs), so all three are compared under the
same protocol.

## Inputs

- `thesis_data/inputs/fixed_design_cases_560.csv`
- `thesis_data/results/radiance_560/sda_ase_560_cases.csv`

The scripts require a complete 560-row simulation summary before training.

## Features

The default feature set (`--features reduced`) has eight inputs, each with its own
physical meaning:

- windowed façades: four 0/1 flags (`has_north_window`, `has_east_window`,
  `has_south_window`, `has_west_window`) derived from `active_facades`;
- `sill_height_m`, `window_width_m`, `window_height_m`, `overhang_depth_m`
  (standardised for the MLP and the linear model; the MLP target is standardised as well).

`--features full` adds `window_type` and `overhang_present` (one-hot encoded). They
are fully determined by `sill_height_m` and `overhang_depth_m` and did not change
the accuracy, so the reduced set is used for the final models.

## Protocol

1. 80/20 split of the 560 cases, stratified by `stratum_id`: each of the 56
   design strata keeps 8 training and 2 test cases.
2. Hyperparameters are tuned separately for each target by randomised search
   (60 configurations) with 5-fold cross-validation on the 448 training cases.
   The folds are stratified by `stratum_id` and identical for all models. The
   configuration with the lowest mean CV RMSE is selected. The linear baseline
   has no hyperparameters; its cross-validation is reported for comparison.
3. The selected configuration is refitted on all training cases, saved, and
   evaluated once on the 112 test cases.
4. Predictions are clipped to 0–100 %, in the CV scores and in the final
   evaluation alike.

The search spaces are defined at the top of each training script.

## Run

Run from the repository root folder with the project environment (see the main README):

```powershell
.\.venv\Scripts\python.exe 3_ml\train_linear.py --overwrite
.\.venv\Scripts\python.exe 3_ml\train_xgboost.py --overwrite
.\.venv\Scripts\python.exe 3_ml\train_mlp.py --overwrite
.\.venv\Scripts\python.exe 3_ml\train_mlp_ensemble.py
```

`train_mlp_ensemble.py` reuses the hyperparameters in `reduced/mlp/best_params.json`, so run it
after `train_mlp.py`.

The figure scripts of the model comparison are in `thesis_data/figure_scripts/` (see the main README).

`--overwrite` replaces an existing output folder; `--n-iter` changes the number of
search configurations; `--features full` uses the ten-feature set. The linear
model takes seconds, XGBoost about 1 minute and the MLP about 3–5 minutes with 8
parallel jobs.

## Outputs

Each run writes `thesis_data/results/ml_training/<feature set>/<model>/` containing:

- `model_<target>.joblib` — the final fitted pipeline (preprocessing + model);
  load it with `joblib.load` in an environment with the package versions listed in
  `run_settings.json`, and call `.predict()` on a DataFrame with the feature columns;
- `metrics.csv` — training and test R², MAE and RMSE of the final models;
- `cv_summary.csv` — mean ± standard deviation of the CV scores of the selected configuration;
- `best_params.json` — selected hyperparameters per target;
- `cv_results_<target>.csv` — every searched configuration with its CV scores, ranked;
- `test_predictions.csv` and `test_predictions.png` — held-out predictions (the `.png` is a quick
  check and is not included in the repository; Fig. 4.16 is drawn by `thesis_data/figure_scripts/fig_model_comparison.py`);
- `train_model_ids.csv` and `test_model_ids.csv` — the fixed split;
- `run_settings.json` — protocol, features, search space, package versions and run time.

XGBoost also writes `feature_importance.csv`.
