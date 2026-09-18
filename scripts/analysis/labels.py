"""Label table: LIMS product sulfur, usable from `available_at`.

Writes `models/labels.parquet` and prints a summary plus the outlier screen.

    python -m scripts.analysis.labels
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import realdata as rd


def build_labels() -> pd.DataFrame:
    lims = rd.load_lims()
    labels = rd.load_lab_sulfur(lims)
    tele = rd.load_telemetry()

    state = rd.running_state(tele)
    pos = tele.index.searchsorted(labels["sampled_at"], side="right") - 1
    labels["running_at_sample"] = [state.iloc[p] if p >= 0 else np.nan for p in pos]

    ends = rd.shutdown_ends(tele)
    since = []
    for t in labels["sampled_at"]:
        p = int(ends.searchsorted(t, side="right")) - 1
        since.append(round((t - ends[p]).total_seconds() / 3600.0, 1) if p >= 0 else np.nan)
    labels["hours_since_restart"] = since

    labels["split"] = np.where(labels["sampled_at"] <= rd.SPLIT_AT, "train", "heldout")
    labels["above_spec"] = labels["sulfur_mg_kg"] > rd.SPEC_LIMIT_MG_KG
    labels["is_outlier"] = labels["sulfur_mg_kg"] > rd.OUTLIER_ABOVE_MG_KG
    return labels


def outlier_screen(labels: pd.DataFrame) -> pd.DataFrame:
    feed = rd.load_feed_sulfur()
    feed["day"] = feed["sampled_at"].dt.normalize()
    rows = []
    for _, r in labels[labels["is_outlier"]].iterrows():
        same_day = feed[feed["day"] == r["sampled_at"].normalize()]
        feed_pct = float(same_day["feed_sulfur_pct"].iloc[0]) if len(same_day) else np.nan
        rows.append({
            "sampled_at": r["sampled_at"],
            "sulfur_mg_kg": r["sulfur_mg_kg"],
            "running_at_sample": r["running_at_sample"],
            "hours_since_restart": r["hours_since_restart"],
            "feed_sulfur_pct_same_day": feed_pct,
            "feed_as_mg_kg": feed_pct * 10_000 if pd.notna(feed_pct) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    labels = build_labels()
    rd.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = rd.MODELS_DIR / "labels.parquet"
    labels.to_parquet(out, index=False)

    running = labels["running_at_sample"]
    print(f"wrote {rd.rel(out)}")
    print(f"lab results            : {len(labels):,}")
    print(f"running at sampling    : {int((running == True).sum()):,}")  # noqa: E712
    print(f"not running            : {int((running == False).sum()):,}")  # noqa: E712
    print(f"running unknown (gaps) : {int(running.isna().sum()):,}")
    print()
    print(labels.groupby("split").agg(
        n=("sulfur_mg_kg", "size"),
        median=("sulfur_mg_kg", "median"),
        above_spec=("above_spec", "sum"),
        outliers=("is_outlier", "sum"),
    ).to_string())
    print()
    screen = outlier_screen(labels)
    print(f"outliers above {rd.OUTLIER_ABOVE_MG_KG:g} mg/kg:")
    print(screen.to_string(index=False) if len(screen) else "  none")


if __name__ == "__main__":
    main()
