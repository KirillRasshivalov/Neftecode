"""Fit the quality baseline: an EWMA of lab results, with an empirical interval.

Why an EWMA and not "same as the last result": consecutive LIMS results are only
weakly autocorrelated, so copying the last one is worse than the training median.
An exponentially weighted mean chosen on the training period beats both.

Leak-free by construction: the prediction for a result sampled at t uses only
results whose `available_at` (sampling + 4 h) is at or before t. The smoothing
factor and the interval are fitted on the **training** period only; the held-out
numbers are evaluation.

Writes `models/quality_baseline.json`.

    python -m scripts.analysis.quality_baseline_fit
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts import realdata as rd

ALPHAS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
WARMUP = 30  # first training results only prime the average


def usable_labels(labels: pd.DataFrame) -> pd.DataFrame:
    keep = (labels["running_at_sample"] == True) & ~labels["is_outlier"]  # noqa: E712
    return labels[keep].sort_values("sampled_at", kind="stable").reset_index(drop=True)


def ewma_after(values: np.ndarray, alpha: float, start: float) -> np.ndarray:
    """EWMA value right after each result has been absorbed."""
    out = np.empty(len(values))
    e = start
    for i, x in enumerate(values):
        e = alpha * x + (1.0 - alpha) * e
        out[i] = e
    return out


def asof_predictions(u: pd.DataFrame, alpha: float, start: float) -> np.ndarray:
    """Prediction for each result from results already available at its sampling time."""
    after = ewma_after(u["sulfur_mg_kg"].to_numpy(), alpha, start)
    cutoff = u["sampled_at"] - rd.LIMS_DELAY
    pos = u["sampled_at"].searchsorted(cutoff, side="right") - 1
    return np.where(pos >= 0, after[np.clip(pos, 0, None)], start)


def main() -> None:
    labels = pd.read_parquet(rd.MODELS_DIR / "labels.parquet")
    u = usable_labels(labels)
    y = u["sulfur_mg_kg"].to_numpy()
    train = (u["split"] == "train").to_numpy()
    held = (u["split"] == "heldout").to_numpy()
    train_fit = train.copy()
    train_fit[np.flatnonzero(train)[:WARMUP]] = False

    median = float(np.median(y[train]))
    grid = {}
    for alpha in ALPHAS:
        pred = asof_predictions(u, alpha, median)
        grid[alpha] = float(np.abs(y[train_fit] - pred[train_fit]).mean())
    alpha = min(grid, key=grid.get)

    pred = asof_predictions(u, alpha, median)
    resid = y[train_fit] - pred[train_fit]
    q05, q50, q95 = (float(np.quantile(resid, q)) for q in (0.05, 0.50, 0.95))
    gaps = u.loc[train, "sampled_at"].diff().dt.total_seconds().div(3600.0)
    ref_gap = float(gaps[(gaps >= 12) & (gaps <= 36)].median())

    yh, ph = y[held], pred[held]
    sse_mean = float(((yh - yh.mean()) ** 2).sum())
    heldout_eval = {
        "n_results": int(held.sum()),
        "mae_ewma": round(float(np.abs(yh - ph).mean()), 4),
        "mae_train_median": round(float(np.abs(yh - median).mean()), 4),
        "r2_ewma": round(1.0 - float(((yh - ph) ** 2).sum()) / sse_mean, 4),
        "r2_train_median": round(1.0 - float(((yh - median) ** 2).sum()) / sse_mean, 4),
        "interval_coverage_p05_p95": round(float(((yh >= ph + q05) & (yh <= ph + q95)).mean()), 4),
        "share_above_spec": round(float((yh > rd.SPEC_LIMIT_MG_KG).mean()), 4),
        "share_p95_below_spec": round(float((ph + q95 < rd.SPEC_LIMIT_MG_KG).mean()), 4),
        "p95_range": [round(float((ph + q95).min()), 3), round(float((ph + q95).max()), 3)],
    }

    params = {
        "model": "ewma_v0",
        "fitted_on": f"train: sampled_at <= {rd.SPLIT_AT.isoformat()}, running, non-outlier",
        "alpha": alpha,
        "alpha_grid_train_mae": {str(a): round(v, 4) for a, v in grid.items()},
        "resid_q05": round(q05, 4),
        "resid_q50": round(q50, 4),
        "resid_q95": round(q95, 4),
        "ref_gap_hours": round(ref_gap, 2),
        "sulfur_median_train": round(median, 4),
        "lims_delay_hours": rd.LIMS_DELAY.total_seconds() / 3600.0,
        "heldout_eval": heldout_eval,
    }
    out = rd.MODELS_DIR / "quality_baseline.json"
    out.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(params, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
