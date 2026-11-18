#!/usr/bin/env python3
"""Five-seed MLP ensemble for prediction uncertainty of annual sDA and ASE.

Every member uses the hyperparameters selected by train_mlp.py
(reduced/mlp/best_params.json) and all 448 training cases of the common split;
only the random seed differs (weight initialisation, minibatch order and the
internal early-stopping hold-out). The spread of the members' predictions is a
model-uncertainty indicator; it does not include the simulation noise of the labels.
"""

import json
import time

import joblib
import numpy as np
import pandas as pd

import common
import train_mlp

SEEDS = [20260924, 20260925, 20260926, 20260927, 20260928]   # member 1 = the selected MLP


def main():
    start = time.perf_counter()
    root = common.OUT_ROOT / "reduced" / "mlp"
    out = common.OUT_ROOT / "reduced" / "mlp_ensemble"
    out.mkdir(parents=True, exist_ok=True)
    data = common.load_dataset(common.DEFAULT_CASES, common.DEFAULT_RESULTS)
    train, test = common.split_dataset(data)                    # same 448 / 112 split
    feats = common.feature_columns("reduced")
    best = json.load(open(root / "best_params.json", encoding="utf-8"))

    rows = test[["model_id"]].copy()
    summary = []
    for target, params in best.items():
        params = {k: tuple(v) if isinstance(v, list) else v for k, v in params.items()}  # JSON list -> tuple
        preds = []
        for seed in SEEDS:
            m = train_mlp.make_estimator("reduced").set_params(**params, regressor__regressor__random_state=seed)
            m.fit(train[feats], train[target])                  # all 448 training cases
            joblib.dump(m, out / f"model_{target}_seed{seed}.joblib")
            p = common.clip(m.predict(test[feats]))
            rows[f"pred_{target}_seed{seed}"] = p
            preds.append(p)
        preds = np.array(preds)
        mean, std = preds.mean(0), preds.std(0, ddof=1)
        actual = test[target].to_numpy()
        rows[f"actual_{target}"] = actual
        rows[f"mean_{target}"], rows[f"std_{target}"] = mean, std

        # member 1 must reproduce the saved selected model
        saved = pd.read_csv(root / "test_predictions.csv", dtype={"model_id": str})
        err = np.abs(mean - actual)
        summary.append({
            "target": target,
            "member1_matches_saved_model": bool(np.allclose(preds[0], saved[f"predicted_{target}"], atol=1e-6)),
            "ensemble_mean_" + "MAE": float(err.mean()),
            "ensemble_mean_RMSE": float(np.sqrt(np.mean((mean - actual) ** 2))),
            "member_MAE_min": float(np.abs(preds - actual).mean(1).min()),
            "member_MAE_max": float(np.abs(preds - actual).mean(1).max()),
            "std_mean": float(std.mean()), "std_median": float(np.median(std)), "std_max": float(std.max()),
            "abs_error_median": float(np.median(err)),
            "share_within_1std": float(np.mean(err <= std)),
            "share_within_2std": float(np.mean(err <= 2 * std)),
            "spearman_std_vs_abs_error": float(pd.Series(std).corr(pd.Series(err), method="spearman")),
        })

    rows.to_csv(out / "ensemble_test_predictions.csv", index=False, float_format="%.6f")
    pd.DataFrame(summary).to_csv(out / "ensemble_summary.csv", index=False, float_format="%.6f")
    (out / "run_settings.json").write_text(json.dumps({
        "members": len(SEEDS), "seeds": SEEDS, "hyperparameters": "reduced/mlp/best_params.json",
        "training_cases": len(train), "test_cases": len(test), "feature_set": "reduced",
        "package_versions": common.package_versions(), "elapsed_s": time.perf_counter() - start,
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary).T.to_string(header=False))
    print(f"Saved to {out}  ({time.perf_counter() - start:.0f}s)")


if __name__ == "__main__":
    main()
