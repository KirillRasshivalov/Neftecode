"""Reference statistics for the reliability baseline, from the training period only.

- operating envelope (p01 / p50 / p99) of each frozen lever on clean operating rows;
- median `242000:T5` per band of `242000:F26`, the basis of the "running hotter than
  usual for this feed rate" deactivation proxy;
- typical feed rate and the lever step sizes.

The envelope is a conservative model range, not an equipment limit: the package states
no operating range, so a historical percentile band stands in for one and is recorded
as an assumption. Writes `models/reliability_reference.json`.

    python -m scripts.analysis.reliability_reference
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from scripts import realdata as rd

F26_BAND = 10.0
MIN_ROWS_PER_BAND = 500  # 10-minute rows: about 3.5 days of operation


def main() -> None:
    tele = rd.load_telemetry()
    clean = tele[rd.clean_operating_mask(tele) & (tele.index <= rd.SPLIT_AT)]

    envelope = {}
    for tag in rd.LEVERS:
        col = clean[rd.tag_id(tag)]
        envelope[rd.tag_id(tag)] = {
            "p01": round(float(col.quantile(0.01)), 4),
            "p50": round(float(col.quantile(0.50)), 4),
            "p99": round(float(col.quantile(0.99)), 4),
        }

    t5, f26 = clean[rd.tag_id("T5")], clean[rd.tag_id("F26")]
    lo_edge = math.floor(envelope[rd.tag_id("F26")]["p01"] / F26_BAND) * F26_BAND
    hi_edge = math.ceil(envelope[rd.tag_id("F26")]["p99"] / F26_BAND) * F26_BAND
    bands = []
    edge = lo_edge
    while edge < hi_edge:
        in_band = t5[(f26 >= edge) & (f26 < edge + F26_BAND)]
        if len(in_band) >= MIN_ROWS_PER_BAND:
            iqr = float(in_band.quantile(0.75) - in_band.quantile(0.25))
            bands.append({
                "lo": edge,
                "hi": edge + F26_BAND,
                "n": int(len(in_band)),
                "median": round(float(in_band.median()), 4),
                "sigma": round(max(iqr / 1.349, 0.5), 4),
            })
        edge += F26_BAND

    # Catalyst cycles, training period only. The excess temperature at which the plant
    # changed the catalyst is the end-of-run level: no equipment data states one.
    train_tele = tele[tele.index <= rd.SPLIT_AT]
    daily = rd.excess_t5_daily(train_tele, bands)
    changes = [c for c in rd.catalyst_changes(daily, rd.shutdown_ends(train_tele))
               if c + pd.Timedelta(days=rd.CATALYST_WINDOW_DAYS) <= rd.SPLIT_AT]
    before = [float(daily.loc[: c - pd.Timedelta(seconds=1)].tail(rd.CATALYST_WINDOW_DAYS).median()) for c in changes]
    catalyst = {
        "changes_in_training": [c.isoformat() for c in changes],
        "excess_before_change_c": [round(v, 2) for v in before],
        "eor_excess_c": round(float(np.median(before)), 2) if before else None,
        "change_fall_c": rd.CATALYST_CHANGE_FALL_C,
    }

    reference = {
        "fitted_on": f"train: date <= {rd.SPLIT_AT.isoformat()}, clean operating rows",
        "catalyst": catalyst,
        "n_rows": int(len(clean)),
        "envelope": envelope,
        "t5_by_f26": bands,
        "typical_f26": envelope[rd.tag_id("F26")]["p50"],
        "steps": {rd.tag_id(tag): float(spec["step"]) for tag, spec in rd.LEVERS.items()},
    }
    out = rd.MODELS_DIR / "reliability_reference.json"
    rd.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {rd.rel(out)}")
    print(json.dumps({k: v for k, v in reference.items() if k != "t5_by_f26"}, ensure_ascii=False, indent=2))
    print(f"t5_by_f26: {len(bands)} bands")
    for b in bands:
        print(f"  F26 [{b['lo']:.0f}, {b['hi']:.0f}): n={b['n']:>6}  T5 median={b['median']:.2f}  sigma={b['sigma']:.2f}")


if __name__ == "__main__":
    main()
