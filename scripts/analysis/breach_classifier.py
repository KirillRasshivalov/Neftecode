"""Breach classifier check: can an off-spec lab result be predicted before it lands?

The regression check (`predictability.py`) asked for the sulfur *level* and found
nothing: held-out R2 ~ 0. This asks the operational question instead — is the lab
result going to be **above 10 mg/kg** — because the signal, if any, may live in the
tail rather than in the middle of the distribution, and because the decision the
operator makes is binary.

Horizons. A result sampled at `t` is predicted from a cutoff at `t - horizon`:

- `0 h`  — the sample is being taken right now. A **nowcast**: it covers the 4 h the
  lab takes, and it is the ceiling of what the instrumentation can say.
- `4 h and beyond` — a **forecast**: the decision is made before the sample exists.
  Only a forecast lets the system act ahead of a bad result. 18 h matches the demo
  cadence: a decision at 16:00, a sample at 10:00 the next morning.

Leakage rules, same discipline as the regression check:

- every feature is computed at the cutoff and uses only telemetry at or before it
  and lab results whose `available_at` (sampling + 4 h) is at or before it, so the
  result being predicted is never among its own inputs;
- rows where the unit was not running at the cutoff are dropped — in production the
  orchestrator refuses there anyway;
- the horizon, feature set, model and regularisation are chosen by expanding-window
  CV **inside the training period**. The held-out period is scored once. The full
  held-out table is printed for transparency, never for selection.

Pre-registered decision rule — the classifier replaces `_prob_above` in
`QualityAgentBaseline` only if **all** of the following hold on the held-out period:

    1. ROC-AUC >= 0.60
    2. the 95 % bootstrap lower bound of that AUC is above 0.50
    3. Brier skill score > 0 (the probabilities beat the base rate)
    4. it beats the probability the quality agent produces today

Three tables are printed, in increasing order of modelling:

    1. the online analyzer as a raw score, with no model at all — the cleanest
       possible evidence, and the one that survives any modelling mistake;
    2. feature groups by horizon, one fixed logistic — says *which* signal carries
       the result and how fast it decays;
    3. the full grid the selection runs on.

Writes `models/breach_classifier.json`.

    python -m scripts.analysis.breach_classifier
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neftecode.agents.quality import _prob_above
from scripts import realdata as rd
from scripts.analysis.quality_baseline_fit import ewma_after, usable_labels

#: Feature assembly lives in `scripts/realdata.py`, shared with the decision path.
CONTEXT = rd.CONTEXT_TAGS
LEVERS = tuple(rd.LEVERS)
PAK = rd.PAK_FEATURE
RATIO = rd.RATIO_ID
LAB_FEATURES = rd.LAB_FEATURE_NAMES
design = rd.asof_matrix

#: Horizons the model selection runs over: a nowcast, the lab delay, and forecasts.
HORIZONS_H = (0, 4, 12, 18, 24)
#: Denser horizons for the cheap diagnostics, to show how fast the signal decays.
DECAY_H = (0, 2, 4, 6, 8, 12, 24)

C_GRID = (0.01, 0.1, 1.0, 10.0)
CV_FOLDS = 5
N_BOOTSTRAP = 2000
SEED = 0

#: Operating point: the threshold is set on the training period to catch this share
#: of breaches, then applied unchanged to the held-out period.
RECALL_TARGET = 0.5

#: Pre-registered acceptance threshold (see the module docstring).
MIN_AUC = 0.60

#: Physically required direction of each feature's effect on P(sulfur > 10).
#: Imposed on the boosting model: a coefficient whose sign contradicts the physics means
#: the model learned the operator's policy, not the process, and it is not shipped.
MONOTONE = {
    rd.tag_id("T5"): -1,     # deeper hydrotreating removes sulfur
    rd.tag_id("T6"): -1,     # same, at the second reactor's inlet
    rd.tag_id("F26"): +1,    # more throughput, less residence time
    rd.tag_id("P13"): -1,    # higher hydrogen partial pressure
    rd.tag_id("Q21"): +1,    # product sulfur analyzer inside the telemetry
    RATIO: -1,               # more recycle gas per unit of throughput
    PAK: +1,                 # the analyzer measures the same quantity
    "lab_ewma": +1,
    "lab_last": +1,
    "lab_above_last": +1,
    "lab_above_share7": +1,
    "lab_hours_since_above": -1,
}


# ---------------------------------------------------------------- features


def agent_probability(history: pd.DataFrame, params: dict) -> np.ndarray:
    """The breach probability `QualityAgentBaseline` reports today for holding the regime."""
    ref_gap = float(params["ref_gap_hours"])
    median = float(params["sulfur_median_train"])
    q05, q95 = float(params["resid_q05"]), float(params["resid_q95"])
    out = np.empty(len(history))
    for i, (mean, age) in enumerate(zip(history["lab_ewma"], history["lab_age_h"])):
        effective = 4.0 * ref_gap if not np.isfinite(age) else float(age)
        base = median if not np.isfinite(mean) else float(mean)
        widen = np.sqrt(max(effective, ref_gap) / ref_gap)
        out[i] = _prob_above(max(base + q05 * widen, 0.0), base + q95 * widen, rd.SPEC_LIMIT_MG_KG)
    return out


# ------------------------------------------------------------------ models


def logit(c: float):
    def fit(x: np.ndarray, y: np.ndarray):
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000)).fit(x, y)
    return fit


def boosting(monotone: list[int]):
    def fit(x: np.ndarray, y: np.ndarray):
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
            early_stopping=False, monotonic_cst=monotone, random_state=SEED,
        ).fit(x, y)
    return fit


def cv_auc(x: np.ndarray, y: np.ndarray, fit) -> float:
    """Expanding-window CV inside the training rows: fit on earlier, score the next block."""
    edges = np.linspace(0, len(y), CV_FOLDS + 2).astype(int)
    found = []
    for k in range(1, CV_FOLDS + 1):
        fit_end, val_end = edges[k], edges[k + 1]
        y_fit, y_val = y[:fit_end], y[fit_end:val_end]
        if fit_end < 50 or val_end - fit_end < 20:
            continue
        if y_fit.min() == y_fit.max() or y_val.min() == y_val.max():
            continue
        model = fit(x[:fit_end], y_fit)
        found.append(float(roc_auc_score(y_val, model.predict_proba(x[fit_end:val_end])[:, 1])))
    return float(np.mean(found)) if found else float("nan")


# ----------------------------------------------------------------- scoring


def scores(y: np.ndarray, p: np.ndarray, base_rate: float) -> dict[str, float]:
    brier = float(brier_score_loss(y, p))
    reference = float(brier_score_loss(y, np.full(len(y), base_rate)))
    return {
        "auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(brier, 5),
        "brier_skill": round(1.0 - brier / reference, 4),
    }


def bootstrap_auc(y: np.ndarray, p: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        out.append(float(roc_auc_score(y[idx], p[idx])))
    return [round(float(np.quantile(out, q)), 4) for q in (0.025, 0.5, 0.975)]


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 5) -> list[dict]:
    edges = np.quantile(p, np.linspace(0.0, 1.0, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi)
        if not mask.any():
            continue
        rows.append({
            "n": int(mask.sum()),
            "predicted": round(float(p[mask].mean()), 4),
            "observed": round(float(y[mask].mean()), 4),
        })
    return rows


def operating_point(
    y_tr: np.ndarray, p_tr: np.ndarray, y_ho: np.ndarray, p_ho: np.ndarray, weeks: float
) -> dict:
    """Threshold set on the training period for RECALL_TARGET, then scored on held-out."""
    positives = np.sort(p_tr[y_tr == 1])[::-1]
    k = max(1, int(round(RECALL_TARGET * len(positives)))) - 1
    threshold = float(positives[k])
    flagged = p_ho >= threshold
    hits = int((flagged & (y_ho == 1)).sum())
    return {
        "threshold": round(threshold, 4),
        "train_recall": round(float((p_tr[y_tr == 1] >= threshold).mean()), 4),
        "heldout_recall": round(hits / max(1, int(y_ho.sum())), 4),
        "heldout_precision": round(hits / max(1, int(flagged.sum())), 4),
        "alarms_per_week": round(int(flagged.sum()) / weeks, 2),
        "false_alarms_per_week": round(int((flagged & (y_ho == 0)).sum()) / weeks, 2),
    }


def linear_parameters(model, names: list[str]) -> dict | None:
    """Scaler and coefficients of a logistic pipeline, so inference needs no sklearn."""
    if not isinstance(model[-1], LogisticRegression):
        return None
    scaler, lr = model[0], model[-1]
    return {
        "features": names,
        "mean": [round(float(v), 6) for v in scaler.mean_],
        "scale": [round(float(v), 6) for v in scaler.scale_],
        "coef": [round(float(v), 6) for v in lr.coef_[0]],
        "intercept": round(float(lr.intercept_[0]), 6),
        "form": "p = 1 / (1 + exp(-(intercept + sum(coef * (x - mean) / scale))))",
    }


# ---------------------------------------------------------------- diagnostics


def analyzer_table(
    windows: pd.DataFrame, times: pd.DatetimeIndex, y: np.ndarray, value: np.ndarray,
    train: np.ndarray, outlier: np.ndarray,
) -> list[dict]:
    """The online analyzer as a raw score, with no model and no fitting at all.

    Reported both with and without the three outliers above 50 mg/kg, because the
    Phase 1 audit read these two sulfur series as uncorrelated and that verdict is
    what three points out of 1 067 do to a Pearson correlation.
    """
    rows = []
    for horizon in DECAY_H:
        x = design(windows, pd.DatetimeIndex(times - pd.Timedelta(hours=horizon)), 0)
        for j, column in enumerate(windows.columns):
            score = x[:, j]
            for split, mask in (("train", train), ("heldout", ~train)):
                ok = mask & np.isfinite(score)
                if ok.sum() < 30 or y[ok].min() == y[ok].max():
                    continue
                trimmed = ok & ~outlier
                rows.append({
                    "horizon_h": horizon,
                    "signal": column,
                    "split": split,
                    "n": int(ok.sum()),
                    "auc_above_spec": round(float(roc_auc_score(y[ok], score[ok])), 4),
                    "spearman_value": round(float(stats.spearmanr(score[ok], value[ok]).statistic), 4),
                    "pearson_value": round(float(stats.pearsonr(score[ok], value[ok]).statistic), 4),
                    "pearson_no_outliers": round(
                        float(stats.pearsonr(score[trimmed], value[trimmed]).statistic), 4
                    ),
                })
    return rows


# -------------------------------------------------------------------- main


def feature_sets(columns: list[str], levers: list[str], context: list[str]) -> dict[str, list[str]]:
    window = [c for c in columns if "|" in c]

    def pick(names: list[str]) -> list[str]:
        return [c for c in window if c.split("|")[0] in names]

    everything = levers + context + [PAK]
    return {
        "lab": list(LAB_FEATURES),
        "levers": pick(levers),
        "levers+lab": pick(levers) + list(LAB_FEATURES),
        "levers+context+pak": pick(everything),
        "all": pick(everything) + list(LAB_FEATURES),
    }


def tag_groups(levers: list[str], context: list[str]) -> dict[str, list[str]]:
    return {
        "pak": [PAK],
        "context": context,
        "levers": levers,
        "levers+context": levers + context,
        "context+pak": context + [PAK],
        "levers+context+pak": levers + context + [PAK],
    }


def main() -> None:
    labels = pd.read_parquet(rd.MODELS_DIR / "labels.parquet")
    params = json.loads((rd.MODELS_DIR / "quality_baseline.json").read_text(encoding="utf-8"))

    running = (
        labels[labels["running_at_sample"] == True]  # noqa: E712
        .sort_values("sampled_at", kind="stable")
        .reset_index(drop=True)
    )
    usable = usable_labels(labels)
    ewma = ewma_after(
        usable["sulfur_mg_kg"].to_numpy(),
        float(params["alpha"]),
        float(params["sulfur_median_train"]),
    )

    y = running["above_spec"].to_numpy().astype(int)
    value = running["sulfur_mg_kg"].to_numpy()
    outlier = running["is_outlier"].to_numpy()
    times = pd.DatetimeIndex(running["sampled_at"])
    train = (running["split"] == "train").to_numpy()
    held_times = times[~train]
    weeks = float((held_times.max() - held_times.min()).total_seconds() / (7 * 86400))

    frame = rd.load_feature_frame()
    telemetry = rd.window_means(frame)
    state = rd.running_state(frame).to_numpy()

    levers = [rd.tag_id(t) for t in LEVERS] + [RATIO]
    context = [rd.tag_id(t) for t in CONTEXT]
    groups = tag_groups(levers, context)

    def at(horizon: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        cutoffs = pd.DatetimeIndex(times - pd.Timedelta(hours=horizon))
        window = pd.DataFrame(design(telemetry, cutoffs, 0), columns=list(telemetry.columns))
        history = rd.lab_history(running, usable, ewma, cutoffs)
        pos = frame.index.searchsorted(cutoffs, side="right") - 1
        on = np.array([bool(state[p]) if p >= 0 and state[p] is True else False for p in pos])
        return pd.concat([window, history], axis=1), on, agent_probability(history, params)

    def run(x: np.ndarray, on: np.ndarray, fit) -> tuple[float, dict, np.ndarray, np.ndarray] | None:
        complete = ~np.isnan(x).any(axis=1) & on
        fit_rows, test_rows = train & complete, ~train & complete
        if y[fit_rows].min() == y[fit_rows].max() or y[test_rows].min() == y[test_rows].max():
            return None
        cv = cv_auc(x[fit_rows], y[fit_rows], fit)
        model = fit(x[fit_rows], y[fit_rows])
        probability = model.predict_proba(x[test_rows])[:, 1]
        row = scores(y[test_rows], probability, float(y[fit_rows].mean()))
        row.update(n_train=int(fit_rows.sum()), n_heldout=int(test_rows.sum()), cv_auc_train=round(cv, 4))
        return cv, row, fit_rows, test_rows

    # --- table 1: the analyzer as a raw score, no model
    analyzer = analyzer_table(rd.window_means(frame[[PAK]]), times, y, value, train, outlier)

    # --- table 2: feature groups by horizon, one fixed logistic
    ablation: list[dict] = []
    for horizon in DECAY_H:
        features, on, _ = at(horizon)
        window = [c for c in features.columns if "|" in c]
        for group, names in groups.items():
            columns = [c for c in window if c.split("|")[0] in names]
            result = run(features[columns].to_numpy(dtype=float), on, logit(1.0))
            if result is None:
                continue
            _, row, _, _ = result
            ablation.append({"horizon_h": horizon, "group": group, "n_features": len(columns), **row})

    # --- table 3: the selection grid
    grid: list[dict] = []
    cache: dict[int, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for horizon in HORIZONS_H:
        cache[horizon] = at(horizon)
        features, on, _ = cache[horizon]
        for variant, names in feature_sets(list(features.columns), levers, context).items():
            x = features[names].to_numpy(dtype=float)
            monotone = [MONOTONE.get(n.split("|")[0], 0) for n in names]
            candidates = {f"logit_C={c:g}": logit(c) for c in C_GRID}
            candidates["boosting"] = boosting(monotone)
            for name, fit in candidates.items():
                result = run(x, on, fit)
                if result is None:
                    continue
                _, row, _, _ = result
                grid.append({
                    "horizon_h": horizon, "variant": variant, "model": name,
                    "n_features": x.shape[1], **row,
                })

    table = pd.DataFrame(grid)
    chosen = table.loc[table["cv_auc_train"].idxmax()]
    horizon, variant, model_name = int(chosen["horizon_h"]), chosen["variant"], chosen["model"]

    # --- the chosen setting, refitted, with the baselines it has to beat
    features, on, agent_p = cache[horizon]
    names = feature_sets(list(features.columns), levers, context)[variant]
    x = features[names].to_numpy(dtype=float)
    complete = ~np.isnan(x).any(axis=1) & on
    fit_rows, test_rows = train & complete, ~train & complete
    monotone = [MONOTONE.get(n.split("|")[0], 0) for n in names]
    fit = boosting(monotone) if model_name == "boosting" else logit(float(model_name.split("=")[1]))
    model = fit(x[fit_rows], y[fit_rows])
    p_tr = model.predict_proba(x[fit_rows])[:, 1]
    p_ho = model.predict_proba(x[test_rows])[:, 1]
    y_tr, y_ho = y[fit_rows], y[test_rows]
    base_rate = float(y_tr.mean())

    prev = features["lab_above_last"].to_numpy()
    rates = {v: float(y_tr[prev[fit_rows] == v].mean()) for v in (0.0, 1.0) if (prev[fit_rows] == v).any()}
    rule = np.array([rates.get(v, base_rate) for v in prev[test_rows]])
    baselines = {
        "base_rate_constant": scores(y_ho, np.full(len(y_ho), base_rate), base_rate),
        "previous_lab_rule": scores(y_ho, rule, base_rate),
        "quality_agent_today": scores(y_ho, agent_p[test_rows], base_rate),
    }

    heldout = scores(y_ho, p_ho, base_rate)
    ci_low, ci_mid, ci_high = bootstrap_auc(y_ho, p_ho)
    checks = {
        "auc_at_least_0.60": heldout["auc"] >= MIN_AUC,
        "bootstrap_lower_above_0.50": ci_low > 0.5,
        "brier_skill_positive": heldout["brier_skill"] > 0,
        "beats_quality_agent_today": heldout["auc"] > baselines["quality_agent_today"]["auc"],
    }
    operating = operating_point(y_tr, p_tr, y_ho, p_ho, weeks)
    linear = linear_parameters(model, names)
    if linear is not None:
        # Travel with the coefficients: the agent quotes the AUC on the card, and the
        # orchestrator's trigger threshold comes from this same run.
        linear["auc_heldout"] = heldout["auc"]
        linear["trigger_threshold"] = operating["threshold"]
    per_tag = None
    if linear is not None:
        coef = pd.Series(linear["coef"], index=names)
        per_tag = (
            coef.groupby([n.split("|")[0] for n in names]).sum()
            .sort_values(key=abs, ascending=False).round(4).to_dict()
        )

    report = {
        "label": "LIMS Mg.Sulfur > 10 mg/kg, Гидроочистка т.о. 2, running at sample, outliers kept as breaches",
        "split_at": rd.SPLIT_AT.isoformat(),
        "n_running_labels": int(len(running)),
        "outliers_kept_as_breaches": int(outlier.sum()),
        "base_rate_train": round(base_rate, 4),
        "base_rate_heldout": round(float(y_ho.mean()), 4),
        "heldout_weeks": round(weeks, 1),
        "selected_by_train_cv": chosen.to_dict(),
        "heldout": heldout,
        "auc_bootstrap_ci95": [ci_low, ci_mid, ci_high],
        "baselines_heldout": baselines,
        "calibration_heldout": calibration(y_ho, p_ho),
        "operating_point": operating,
        "acceptance_checks": checks,
        "accepted": all(checks.values()),
        "summed_standardised_coefficients": per_tag,
        "linear_model": linear,
        "analyzer_raw_score": analyzer,
        "feature_group_ablation": ablation,
        "all_runs": grid,
    }
    out = rd.MODELS_DIR / "breach_classifier.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    pd.set_option("display.width", 220)
    print(f"wrote {rd.rel(out)}\n")
    print(
        f"labels: {len(running)} running, base rate train {base_rate:.3f} / held-out {y_ho.mean():.3f}, "
        f"{int(y_ho.sum())} held-out breaches over {weeks:.1f} weeks\n"
    )

    raw = pd.DataFrame(analyzer)
    print("1. online analyzer as a raw score, no model, no fitting")
    print(raw.pivot(index=["signal", "split"], columns="horizon_h", values="auc_above_spec").to_string())
    print("\n   correlation with the lab value at horizon 0 (pearson with / without the 3 outliers):")
    for _, r in raw[raw["horizon_h"] == 0].iterrows():
        print(f"     {r['signal']:<18} {r['split']:<8} spearman {r['spearman_value']:+.3f}   "
              f"pearson {r['pearson_value']:+.3f} -> {r['pearson_no_outliers']:+.3f} without outliers")

    print("\n2. held-out AUC by feature group and horizon (logistic, C=1)")
    print(pd.DataFrame(ablation).pivot(index="group", columns="horizon_h", values="auc").to_string())
    print("\n   the same, train CV AUC - what the selection actually sees")
    print(pd.DataFrame(ablation).pivot(index="group", columns="horizon_h", values="cv_auc_train").to_string())

    print("\n3. selection grid, top 12 by train CV")
    print(table.sort_values("cv_auc_train", ascending=False).head(12)[
        ["horizon_h", "variant", "model", "n_features", "cv_auc_train", "auc", "pr_auc",
         "brier_skill", "n_train", "n_heldout"]
    ].to_string(index=False))
    print("\n   mean held-out AUC by horizon:")
    print(table.groupby("horizon_h")["auc"].agg(["mean", "min", "max"]).round(4).to_string())

    print(f"\nselected by training CV: horizon {horizon} h, {variant}, {model_name}")
    print(f"held-out: {heldout}")
    print(f"AUC 95 % bootstrap CI: {ci_low} .. {ci_high} (median {ci_mid})")
    print("\nbaselines on the same held-out rows:")
    for name, scored in baselines.items():
        print(f"  {name:<22} {scored}")
    if per_tag:
        print("\nsummed standardised coefficients per tag:")
        for tag, coefficient in per_tag.items():
            print(f"  {tag:<24} {coefficient:+.3f}")
    print("\ncalibration (held-out, equal-count bins):")
    for row in report["calibration_heldout"]:
        print(f"  n={row['n']:<4} predicted {row['predicted']:.3f}  observed {row['observed']:.3f}")
    print(f"\noperating point: {report['operating_point']}")
    print("\nacceptance checks (pre-registered):")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\nACCEPTED: {report['accepted']}")


if __name__ == "__main__":
    main()
