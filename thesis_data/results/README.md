# Results: what each folder and file contains

All values are stored results of the thesis; nothing here has to be recomputed. sDA = sDA300/50%,
ASE = ASE1000,250h, both in % of the floor area; errors in percentage points (pp). Model files
(`.joblib`) are loaded by the code and are not meant to be opened by hand.

## `radiance_560/` — Radiance results of the 560 sampled designs (training and test data)

| File | Content |
|---|---|
| `sda_ase_560_cases.csv` | **The dataset**: one row per design with its inputs, sDA and ASE (the labels used to train the models) |
| `run_info.json` | Run time and number of workers of the last simulation run. `models_run` is 557 because the script skips designs whose `metrics.json` already exists: three designs had been simulated in an earlier run. All 560 designs have results. |
| `model_NNN/metrics.json` | sDA, ASE and run time of one design |
| `model_NNN/da300_map.npy` | 32 × 32 map: share of working hours with ≥ 300 lx at each sensor (0–1); sDA = share of points ≥ 0.5 |
| `model_NNN/ase_hours_map.npy` | 32 × 32 map: working hours with direct sun ≥ 1000 lx at each sensor; ASE = share of points > 250 h |
| `model_NNN/room.rad` | Radiance scene of the design (room, windows, overhang) |

Map rows run from south to north, columns from west to east. Design inputs of every `model_NNN`
are in `thesis_data/inputs/fixed_design_cases_560.csv`.

## `ml_training/reduced/` — trained models (thesis Section 4.3)

`reduced` is the eight-input feature set used for all final models.

| Folder | Content |
|---|---|
| `linear/`, `xgboost/`, `mlp/` | The three compared models (Tables 4.11–4.13, Fig. 4.16): `model_<target>.joblib` (final fitted model), `metrics.csv` (train/test R², MAE, RMSE), `cv_summary.csv` (5-fold CV of the selected configuration), `best_params.json` (selected hyperparameters), `test_predictions.csv` (112 test cases), `run_settings.json` (search space, protocol and package versions) |
| `linear/model_<target>.joblib` | The coefficients of the linear models are Table 4.10 |
| `xgboost/cv_results_<target>.csv`, `mlp/cv_results_<target>.csv` | Every configuration of the hyperparameter search with its CV scores (Fig. 4.14, 4.15) |
| `mlp/model_<target>.joblib` | The selected single MLP (first member of the ensemble; used for Fig. 4.25, 4.26) |
| `mlp/train_model_ids.csv`, `mlp/test_model_ids.csv` | The fixed 448 / 112 split, the same for all models |
| `mlp/heldout_predictions_cv_and_test.csv` | Prediction of every one of the 560 designs by a model that did not see it (Fig. 4.25) |
| `mlp/limitations_numbers.json` | Error numbers quoted in Section 4.3.6 (e.g. 13 cases with ASE > 70 %, under-predicted by 0.76 pp) |
| `mlp_ensemble/` | **The surrogate used by the workflow**: five MLPs per target (`model_<target>_seed<seed>.joblib`); `ensemble_test_predictions.csv` (each member, mean and spread on the 112 test cases); `ensemble_summary.csv` (Table 4.14); `error_margins.csv` (Table 4.18); `decision_config.json` (90 % error margins and spread levels used by the decision rule, Tables 4.18–4.19) |
| `map_ensemble/` | Map surrogate used by the workflow: PCA patterns (`pca_<map>.joblib`) and five MLPs per map predicting the pattern weights; `map_metrics.csv` (accuracy on the 112 test cases) |
| `figures/` | Figures 4.14–4.16, 4.25, 4.26, 4.28, 4.29 (drawn by `thesis_data/figure_scripts/`) |

## `extensions/` — supporting analyses of the thesis

| Folder | Thesis | Main files |
|---|---|---|
| `data_distribution/` | Fig. 4.9, Section 4.3.3 | `fig_data_distribution.png`; `statistics_560_cases.json` (ranges, medians and counts per LEED level of the 560 results) |
| `grid_convergence/` | Fig. 4.6 | `grid_convergence.csv`: error of sDA/ASE counted on coarser grids against the 32 × 32 grid |
| `learning_curve/` | Fig. 4.17 | `learning_curve_summary.csv` (test RMSE per training size, mean of 5 repeats), `learning_curve_runs.csv` (every repeat), `fig_learning_curve.png` (Fig. 4.17, drawn by `thesis_data/figure_scripts/fig_learning_curve.py`) |
| `single_vs_ensemble/` | Fig. 4.18, Table 4.14 | `single_vs_ensemble_metrics.csv` and figure |
| `pca_map_da300_10x10/` (DA300), `pca_map_ase_10x10/` (direct-sun hours) | Fig. 4.19–4.24, Table 4.15 | `pca_k_cv.csv` (CV error per number of patterns k), `pca_k_comparison.csv` (test errors per k, PCA vs point-wise MLP), `pca_explained_variance.csv`, `*_maps_10x10.csv` (simulated 10 × 10 maps of the 560 cases), `test_predicted_maps_10x10.csv` (PCA + MLP) and `..._direct.csv` (point-wise MLP), `summary.json`, figures |
| `pca_map_32/da300/`, `pca_map_32/ase/` | Table 4.16 | `metric_comparison.csv` (sDA/ASE counted from predicted 32 × 32 maps vs direct ensemble), `test_metric_predictions.csv` (per test case), `pca_k_cv.csv`, `summary.json` |
| `overhang_gap/` | Fig. 4.31 | `overhang_gap_radiance.csv` (Radiance and MLP for 12 overhang depths of one design), and the figure |
| `independent_validation/` | Section 5.1, Fig. 5.1–5.2, Tables 5.1–5.2 | `test_protocol.json` (rules fixed before simulating), `designs_and_predictions_before_simulation.json` (the 147 new designs of groups A, B, C with their MLP predictions), `predicted_vs_radiance_147_designs.csv` (prediction and Radiance result of each design), `accuracy_and_coverage_by_group.json` (MAE, RMSE, margin coverage per group), figures of groups A and C |
| `workflow_examples/` | Section 5.2, Appendix A | `A046_tool_result.json`, `B003_tool_result.json` (full output of the MCP tool `evaluate_design`), `<id>_report.txt` (report as returned), `<id>_report_appendix.md` (Appendix A version), map figures, and `B003_radiance_run/` (the Radiance simulation the tool started for B003) |

`B003_radiance_run/` has the same files as a `radiance_560/model_NNN/` folder; its maps are drawn
in Fig. 5.5, and `B003_radiance_run_case.json` holds the design inputs. The individual Radiance runs of
the overhang check and of the independent test are not included: their sDA and ASE are in
`overhang_gap_radiance.csv` and `predicted_vs_radiance_147_designs.csv`.
