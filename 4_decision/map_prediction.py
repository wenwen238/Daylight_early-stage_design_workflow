"""Advisory output of an accepted MLP evaluation: predicted 10 x 10 maps.

The maps do not change the decision of decision.py, which uses only sDA and ASE of the
32 x 32 grid.

predict_maps(X)
    DA300 map [%] and map of occupied hours with direct sun above 1000 lx [h] at the 10 x 10
    cell centres x, y = 0.3, 0.9, ..., 5.7 m (rows from south to north, columns from west to east).
    PCA + MLP of 3_ml/train_map_ensemble.py: the five MLPs predict the PCA scores, their mean is
    reconstructed to a map and limited to the valid range.
"""

import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

MAP_DIR = config.ML_TRAINING / "reduced" / "map_ensemble"
MAP_SEEDS = [20260924, 20260925, 20260926, 20260927, 20260928]   # as train_map_ensemble.py
MAPS = {"da300": ((0.0, 100.0), "%"), "ase_hours": ((0.0, 3650.0), "h")}
MAP_N = 10
MAP_POINTS_M = [round(0.3 + 0.6 * i, 1) for i in range(MAP_N)]
MAP_ACCURACY = {"da300": "test RMSE per point 5.0 pp", "ase_hours": "test RMSE per point 17 h"}


@lru_cache(maxsize=1)
def _map_models():
    return {name: (joblib.load(MAP_DIR / f"pca_{name}.joblib"),
                   [joblib.load(MAP_DIR / f"model_{name}_scores_seed{s}.joblib") for s in MAP_SEEDS])
            for name in MAPS}


def predict_maps(X):
    out = {"points_m": MAP_POINTS_M, "orientation": "rows from south (y = 0.3 m) to north, columns from west to east",
           "note": "predicted maps (PCA + MLP), advisory only: the decision uses sDA and ASE of the 32 x 32 grid"}
    for name, (pca, members) in _map_models().items():
        valid, unit = MAPS[name]
        scores = np.mean([m.predict(X).reshape(len(X), -1) for m in members], axis=0)
        p = np.clip(pca.inverse_transform(scores), *valid).reshape(MAP_N, MAP_N)
        out[name] = {"unit": unit, "accuracy": MAP_ACCURACY[name], "values": np.round(p, 1).tolist()}
    return out

