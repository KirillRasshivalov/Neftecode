"""Predictability check: does 24-2000 telemetry carry signal about lab sulfur?

Trains the simplest honest model, ridge regression on lagged window means, on the
training period and evaluates it on the held-out period against the training median
and the EWMA baseline the quality agent uses today. A lag sweep estimates the dead
time. A gradient-boosting fit at the selected setting checks that a flat result is
not just a linearity artefact. Lever signs are checked against the physics, feed first.

Leakage rules:
- features for a lab result sampled at t use telemetry at or before t only;
- the regularisation strength is chosen by expanding-window CV inside the training
  period, and the feature set and lag are selected by that CV score, never by
  held-out results. The full held-out table is printed for transparency only.

Writes `models/predictability.json`.

    python -m scripts.analysis.predictability
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts import realdata as rd
from scripts.analysis.quality_baseline_fit import asof_predictions, usable_labels

LAGS_H = (0, 2, 4, 8, 12, 24)
LAMBDAS = (0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)
CV_FOLDS = 5

LEVERS = tuple(rd.LEVERS)
#: Feature assembly lives in `scripts/realdata.py`, so the fitting scripts and the
#: decision path share one definition. Re-exported here for readability.
CONTEXT = rd.CONTEXT_TAGS
RATIO = rd.RATIO_ID
PAK = rd.PAK_FEATURE

#: Physically expected direction of each lever's effect on sulfur.
EXPECTED_SIGN = {rd.tag_id("F26"): +1, rd.tag_id("T5"): -1, rd.tag_id("P13"): -1, RATIO: -1}


#: Feature rows as of a timestamp minus a lag; `scripts/realdata.py` owns the rule.
design = rd.asof_matrix
window_means = rd.window_means


# ------------------------------------------------------------------- ridge


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> tuple:
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd == 0] = 1.0
    z = (x - mu) / sd
    y_mean = y.mean()
    beta = np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ (y - y_mean))
    return mu, sd, y_mean, beta


def ridge_predict(model: tuple, x: np.ndarray) -> np.ndarray:
    mu, sd, y_mean, beta = model
    return y_mean + ((x - mu) / sd) @ beta


def cv_lambda(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Expanding-window CV inside the training rows: fit on earlier, validate on the next block."""
    edges = np.linspace(0, len(y), CV_FOLDS + 2).astype(int)
    scores = {}
    for lam in LAMBDAS:
        errors = []
        for k in range(1, CV_FOLDS + 1):
            fit_end, val_end = edges[k], edges[k + 1]
            if fit_end < 50 or val_end - fit_end < 20:
                continue
            model = ridge_fit(x[:fit_end], y[:fit_end], lam)
            errors.append(float(np.mean((y[fit_end:val_end] - ridge_predict(model, x[fit_end:val_end])) ** 2)))
        scores[lam] = float(np.mean(errors))
    best = min(scores, key=scores.get)
    return best, scores[best]


def r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


# -------------------------------------------------------------------- main


def main() -> None:
    labels = pd.read_parquet(rd.MODELS_DIR / "labels.parquet")
    params = json.loads((rd.MODELS_DIR / "quality_baseline.json").read_text(encoding="utf-8"))
    u = usable_labels(labels)
    y = u["sulfur_mg_kg"].to_numpy()
    times = pd.DatetimeIndex(u["sampled_at"])
    train = (u["split"] == "train").to_numpy()
    held = ~train
    ewma = asof_predictions(u, float(params["alpha"]), float(params["sulfur_median_train"]))
    median = float(np.median(y[train]))

    features = rd.window_means(rd.load_feature_frame())

    def cols(names: list[str]) -> list[str]:
        return [c for c in features.columns if c.split("|")[0] in names]

    lever_names = [rd.tag_id(t) for t in LEVERS] + [RATIO]
    context_names = lever_names + [rd.tag_id(t) for t in CONTEXT]
    variants = {
        "levers": cols(lever_names),
        "levers+context": cols(context_names),
        "levers+context+pak": cols(context_names + [PAK]),
    }

    rows = []
    for variant, names in variants.items():
        frame = features[names]
        for lag in LAGS_H:
            x = design(frame, times, lag)
            complete = ~np.isnan(x).any(axis=1)
            fit_rows, test_rows = train & complete, held & complete
            lam, cv_mse = cv_lambda(x[fit_rows], y[fit_rows])
            model = ridge_fit(x[fit_rows], y[fit_rows], lam)
            pred = ridge_predict(model, x[test_rows])
            yt = y[test_rows]
            rows.append({
                "variant": variant,
                "lag_h": lag,
                "n_features": x.shape[1],
                "lambda": lam,
                "cv_mse_train": round(cv_mse, 4),
                "n_train": int(fit_rows.sum()),
                "n_heldout": int(test_rows.sum()),
                "r2_train": round(r2(y[fit_rows], ridge_predict(model, x[fit_rows])), 4),
                "r2_heldout": round(r2(yt, pred), 4),
                "mae_heldout": round(mae(yt, pred), 4),
                "r2_median": round(r2(yt, np.full(len(yt), median)), 4),
                "mae_median": round(mae(yt, np.full(len(yt), median)), 4),
                "r2_ewma": round(r2(yt, ewma[test_rows]), 4),
                "mae_ewma": round(mae(yt, ewma[test_rows]), 4),
            })
    table = pd.DataFrame(rows)

    # Selection by training CV only.
    chosen = table.loc[table["cv_mse_train"].idxmin()]
    lever_rows = table[table["variant"] == "levers"]
    lever_choice = lever_rows.loc[lever_rows["cv_mse_train"].idxmin()]

    # Sign check on the levers-only model at its CV-chosen lag.
    x = design(features[variants["levers"]], times, int(lever_choice["lag_h"]))
    complete = ~np.isnan(x).any(axis=1)
    fit_rows = train & complete
    _, _, _, beta = ridge_fit(x[fit_rows], y[fit_rows], float(lever_choice["lambda"]))
    per_tag: dict[str, float] = {}
    for name, coef in zip(variants["levers"], beta):
        per_tag[name.split("|")[0]] = per_tag.get(name.split("|")[0], 0.0) + float(coef)
    signs = {
        tag: {
            "sum_std_coef": round(value, 4),
            "expected": {1: "+", -1: "-"}.get(EXPECTED_SIGN.get(tag), "n/a"),
            "matches": None if tag not in EXPECTED_SIGN else bool(np.sign(value) == EXPECTED_SIGN[tag]),
        }
        for tag, value in per_tag.items()
    }

    # Nonlinear check at the chosen setting.
    gbm = None
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor

        x = design(features[variants[chosen["variant"]]], times, int(chosen["lag_h"]))
        complete = ~np.isnan(x).any(axis=1)
        fit_rows, test_rows = train & complete, held & complete
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, random_state=0
        ).fit(x[fit_rows], y[fit_rows])
        pred = model.predict(x[test_rows])
        gbm = {
            "variant": chosen["variant"],
            "lag_h": int(chosen["lag_h"]),
            "r2_train": round(r2(y[fit_rows], model.predict(x[fit_rows])), 4),
            "r2_heldout": round(r2(y[test_rows], pred), 4),
            "mae_heldout": round(mae(y[test_rows], pred), 4),
        }
    except ImportError:
        pass

    report = {
        "label": "LIMS Mg.Sulfur, Гидроочистка т.о. 2, running, non-outlier",
        "split_at": rd.SPLIT_AT.isoformat(),
        "selected_by_train_cv": chosen.to_dict(),
        "levers_only_selected_by_train_cv": lever_choice.to_dict(),
        "lever_signs": signs,
        "gradient_boosting_check": gbm,
        "all_runs": rows,
    }
    out = rd.MODELS_DIR / "predictability.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(f"wrote {rd.rel(out)}\n")
    print(table[["variant", "lag_h", "lambda", "cv_mse_train", "r2_train", "r2_heldout", "mae_heldout",
                 "mae_median", "mae_ewma", "n_heldout"]].to_string(index=False))
    print("\nselected by training CV:", {k: chosen[k] for k in ("variant", "lag_h", "lambda", "r2_heldout", "mae_heldout", "mae_ewma")})
    print("levers only, selected by training CV:", {k: lever_choice[k] for k in ("lag_h", "r2_heldout", "mae_heldout")})
    print("lever signs:", json.dumps(signs, ensure_ascii=False))
    print("gradient boosting check:", gbm)


if __name__ == "__main__":
    main()
