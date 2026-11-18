# Daylight surrogate workflow: Radiance + MLP ensemble + MCP/Rhino

Final workflow of the thesis. It predicts annual sDA300/50% and ASE1000,250h for a
6 × 6 × 3 m office room in Milan, decides between the MLP prediction and a Radiance
simulation (thesis Section 4.4), and draws the design and its result in Rhino
through MCP.

```
1_sampling/   560 fixed design cases (stratified LHS)        -> thesis_data/inputs/fixed_design_cases_560.*
2_radiance/   annual sDA/ASE with Radiance for the 560 cases -> thesis_data/results/radiance_560/
3_ml/         linear / XGBoost / MLP training, 5-seed MLP ensemble, PCA + MLP map ensemble
                                                             -> thesis_data/results/ml_training/reduced/
4_decision/   coverage check -> ensemble mean -> ±90% margin vs LEED thresholds -> MLP or Radiance;
              design suggestions (suggest.py)
5_mcp/        mcp_server.py (started by Claude Code), report.py (fixed report of the tools) and
              rhino_bridge.py (run inside Rhino)
6_extensions/ supporting analyses of the thesis (grid convergence, learning curve, map
              prediction, overhang gap, independent test, workflow examples)
                                                             -> thesis_data/results/extensions/
thesis_data/
  inputs/          the 560 sampled designs and the Milan weather data
  results/         all stored results: Radiance, trained models, supporting analyses
  figure_scripts/  scripts that draw the thesis figures from the saved results (no retraining)
config.py     all paths and the Radiance location
```

All results of the thesis are included (Radiance results of the 560 cases, the trained
models and the results of the supporting analyses), so nothing has to be recomputed to use
the decision rule or MCP, or to redraw the figures. [`thesis_data/README.md`](thesis_data/README.md)
explains the three folders; what each data and result file contains is described in
[`thesis_data/inputs/README.md`](thesis_data/inputs/README.md) and
[`thesis_data/results/README.md`](thesis_data/results/README.md).

## Thesis figures and analyses

Folders 1–5 are the workflow; `6_extensions/` and `thesis_data/figure_scripts/` only reproduce the
supporting analyses and figures of the thesis and are not needed to use the workflow.

| Thesis | Calculation | Figure script (`thesis_data/figure_scripts/`) | Results |
|---|---|---|---|
| Fig. 4.5 input distribution | — | `fig_data_distribution.py` | `thesis_data/results/extensions/data_distribution/` |
| Fig. 4.6 grid convergence | `6_extensions/grid_convergence.py` | `fig_grid_convergence.py` | `thesis_data/results/extensions/grid_convergence/` |
| Fig. 4.14–4.16, Tables 4.10–4.12 model comparison | `3_ml/train_linear.py`, `train_xgboost.py`, `train_mlp.py` | `fig_hyperparameter_search.py`, `fig_model_comparison.py` | `thesis_data/results/ml_training/reduced/` |
| Fig. 4.17 learning curve | `6_extensions/learning_curve.py` | (drawn by the calculation) | `thesis_data/results/extensions/learning_curve/` |
| Fig. 4.18, Table 4.13 single MLP vs ensemble | `3_ml/train_mlp_ensemble.py` | `fig_single_vs_ensemble.py` | `thesis_data/results/extensions/single_vs_ensemble/` |
| Fig. 4.19–4.24, Table 4.14 map prediction (10 × 10) | `6_extensions/pca_map_prediction.py --metric da300` / `--metric ase`, `3_ml/train_map_ensemble.py` | (drawn by the calculation) | `thesis_data/results/extensions/pca_map_da300_10x10/`, `pca_map_ase_10x10/`, `thesis_data/results/ml_training/reduced/map_ensemble/` |
| Table 4.15 sDA/ASE counted from 32 × 32 maps | `6_extensions/pca_map_32.py` | — | `thesis_data/results/extensions/pca_map_32/` |
| Fig. 4.25, 4.26, Section 4.3.6 numbers | (held-out predictions computed by the figure script) | `fig_mlp_limitations.py` | `thesis_data/results/ml_training/reduced/mlp/limitations_numbers.json` |
| Fig. 4.28, 4.29, Tables 4.17, 4.18 margins and spread | `4_decision/decision.py calibrate` | `fig_ensemble_uncertainty.py` | `thesis_data/results/ml_training/reduced/mlp_ensemble/` |
| Fig. 4.31 overhang 0–0.30 m | `6_extensions/overhang_gap_check.py` | (drawn by the calculation) | `thesis_data/results/extensions/overhang_gap/` |
| Section 5.1 independent test (Fig. 5.1, group C, Tables 5.1, 5.2) | `6_extensions/independent_validation.py` | `fig_independent_groupC.py` | `thesis_data/results/extensions/independent_validation/` |
| Section 5.2 and Appendix A, designs A046 and B003 | `6_extensions/workflow_examples_independent.py`, `workflow_examples_reports.py` | `fig_workflow_examples_maps.py` | `thesis_data/results/extensions/workflow_examples/` |

Run every script from this folder, e.g. `.\.venv\Scripts\python.exe thesis_data/figure_scripts/fig_model_comparison.py`.
Figures of the ML step are written to `thesis_data/results/ml_training/reduced/figures/`. The Radiance run of
B003 made by the MCP tool is stored in `thesis_data/results/extensions/workflow_examples/B003_radiance_run/`.

## Requirements

| What | Needed for | Install |
|---|---|---|
| Python 3.12 | everything | python.org |
| Python packages | everything | `requirements.txt` (below) |
| Radiance 5.x | Radiance simulations (step 2 and MCP `run_radiance`) | radiance-online.org; default location `C:\Radiance`, otherwise set `RADIANCE_ROOT` |
| Rhino 8 | drawing only | optional: without Rhino, prediction and simulation still work |
| Claude Code (or another MCP client) | using the workflow through chat | optional |

## Set up (once)

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If Radiance is not in `C:\Radiance`:

```powershell
setx RADIANCE_ROOT "D:\path\to\Radiance"
```

## Use it through MCP (Claude Code + Rhino)

1. Open Rhino 8, run `ScriptEditor`, open `5_mcp\rhino_bridge.py` and click **Run**.
   Wait for `Bridge v2 started.` If it says port 8765 is still in use, an older bridge is
   running: close and reopen Rhino, then run the script again.
2. Start Claude Code in this folder. `.mcp.json` starts `5_mcp\mcp_server.py`
   automatically; approve the `rhino` server when asked.
3. Ask in chat, for example: "Evaluate a south window, conventional, 1.8 m wide,
   1.5 m high, with a 0.6 m overhang", or "Show model_040 in Rhino".

MCP tools of the workflow:

| Tool | What it does |
|---|---|
| `design_space_info` | valid inputs, LEED thresholds, error margins, spread-level boundaries |
| `evaluate_design` | full decision rule; runs Radiance automatically when needed (30–90 s); draws in Rhino. For an accepted MLP prediction it also returns, as advisory output, predicted 10 × 10 DA300 and direct-sun-hour maps |
| `suggest_changes` | advisory design suggestions (`4_decision/suggest.py`): the closest designs of the sampled design space that reach the next LEED level robustly, found by predicting all 430 976 designs on a 0.05 m grid with the MLP ensemble; optional Radiance check of the first suggestion |
| `run_radiance` | Radiance simulation of any buildable design, e.g. to confirm a high-spread prediction |
| `draw_design` | draw a design in Rhino without evaluating it |
| `get_sample_case` | one of the 560 cases with its stored Radiance result and the rule's decision |

`evaluate_design`, `run_radiance`, `get_sample_case` and `suggest_changes` start their result with `report`:
a fixed report built by `5_mcp/report.py` from the calculated values, which the server instructs the assistant
to show unchanged (1 Result, 2 Design inputs, 3 Basis of the result, 4 Daylight maps, 5 Notes, 6 Next step).

The general Rhino tools (`create_box`, `create_layer`, `call_rhino_tool`, ...) are also available.
Designs are drawn on layers `Daylight::*`; the ceiling layer is hidden so the DA300 map on the
work plane is visible from above. When maps are available, a copy of the room 8 m to the east
shows the map of direct-sun hours (ASE, cyan line = 250 h). For an accepted MLP prediction both
maps are the predicted 10 × 10 maps from PCA + MLP (`3_ml/train_map_ensemble.py`, `4_decision/map_prediction.py`); they
are advisory and do not change the decision, which uses only sDA and ASE.

### Design parameters

- `active_facades`: windowed façades, any of `north`, `east`, `south`, `west`
- `window_type`: `conventional` (sill 0.90 m, give `window_height_m`) or `floor_to_ceiling` (sill 0 m, 3.00 m)
- `window_width_m`; `overhang_depth_m` (0 = no overhang)

The same window is placed, centred, on every active façade; an overhang sits at the window head
and is as long as the window is wide (the sampling rules). Designs inside the sampled design
space (14 façade patterns, width 0.60–2.40 m, conventional height 0.60–2.10 m, overhang 0 or
0.30–1.50 m) can be predicted; other buildable designs go directly to Radiance.

## Run the steps by hand

```powershell
# decision rule
.\.venv\Scripts\python.exe 4_decision\decision.py check-test     # thesis numbers on the 112 test cases
.\.venv\Scripts\python.exe 4_decision\decision.py evaluate --facades south --window-type conventional --width 1.8 --height 1.5 --overhang 0.6
.\.venv\Scripts\python.exe 4_decision\decision.py calibrate      # only after retraining the ensemble

# Radiance: missing cases only / chosen cases / everything again
.\.venv\Scripts\python.exe 2_radiance\annual_metrics.py --workers 8
.\.venv\Scripts\python.exe 2_radiance\annual_metrics.py --models model_040 model_418 --force
.\.venv\Scripts\python.exe 2_radiance\annual_metrics.py --force --workers 8      # ~40 min with 8 workers

# ML (see 3_ml\README.md)
.\.venv\Scripts\python.exe 3_ml\train_mlp.py --overwrite
.\.venv\Scripts\python.exe 3_ml\train_mlp_ensemble.py
.\.venv\Scripts\python.exe 3_ml\train_map_ensemble.py      # predicted DA300 / direct-sun-hour maps
```

`1_sampling\final_sampling.py` refuses to overwrite the frozen 560 cases unless `--overwrite` is given.

## Notes

- Radiance's sky calculation is stochastic (ambient sampling): rerunning the same case gives
  slightly different sDA values (ASE, a direct-sun calculation, is identical).
- Retraining a model can also give slightly different numbers than the stored ones (for example
  `6_extensions/pca_map_32.py`: sDA MAE 1.39 instead of 1.34 pp), although the seeds are fixed.
  The stored results in `thesis_data/results/` are the ones reported in the thesis.
- After retraining the ensemble, run `decision.py calibrate` so the margins and spread levels
  are derived from the new test predictions (`thesis_data/results/ml_training/reduced/mlp_ensemble/decision_config.json`).
- `thesis_data/results/radiance_common/` (sky matrix, sun sources) and `thesis_data/results/mcp_runs/` (MCP simulations)
  are created automatically and can be deleted; they are not part of the repository.
- The saved `.joblib` models must be loaded with the package versions in `requirements.txt`
  (scikit-learn 1.9.1, Python 3.12). Radiance 5.4a was used for all simulations.
