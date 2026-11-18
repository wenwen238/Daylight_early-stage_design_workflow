# thesis_data

Everything the thesis results are made of, next to the workflow code in folders `1_sampling/` to
`6_extensions/`. Nothing here has to be recomputed: the code reads these files directly.

| Folder | What it holds | Details |
|---|---|---|
| [`inputs/`](inputs/) | **Input data.** The 560 sampled façade designs and the Milan weather file used by Radiance. | [`inputs/README.md`](inputs/README.md) |
| [`results/`](results/) | **Stored results.** Radiance results of the 560 designs (the training data), the trained models used by the workflow, and the results of the supporting analyses (grid convergence, learning curve, map prediction, overhang check, independent test, workflow examples). | [`results/README.md`](results/README.md) lists every folder and file and the thesis figure or table it belongs to |
| [`figure_scripts/`](figure_scripts/) | **Figure scripts.** Python scripts that redraw thesis figures from the stored results, without retraining or simulating. They write the figures into `results/`. | Table "Thesis figures and analyses" in the [main README](../README.md) |

Run every script from the repository root, for example:

```powershell
.\.venv\Scripts\python.exe thesis_data\figure_scripts\fig_model_comparison.py
```

Some thesis figures are drawn directly by the calculation scripts in `6_extensions/` instead of a
separate figure script; the main README shows which.
