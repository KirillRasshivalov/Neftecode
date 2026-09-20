"""Reference statistics for the throughput and energy proxies, training period only.

The package contains no economic data (`docs/data_audit.md` §7.2), so the cost side of
the ranking is an **energy proxy in arbitrary units**, which the task statement allows
("энергозатраты и/или прозрачный стоимостной прокси"). It has two terms, both built from
levers whose physical quantity the corrected reference states:

- **fired heater** — duty to bring the gas/feed mixture to reaction temperature. Per unit
  of feed the duty is proportional to the temperature rise, so the term is
  `(T5 - heat_reference_c)`;
- **recycle compressor** — work to circulate the gas. Per unit of *product* the work is
  proportional to the gas-to-feed ratio times the discharge pressure, so the term is
  `(F2 / F26) * P13`.

Each term is divided by its value at the training-period median regime, so the index is
1.0 there and a candidate's effect reads as a percentage. `W_HEAT` / `W_COMPRESSION` are
the assumed split of hydrotreater energy between the two consumers.

`heat_reference_c` is an assumption: the package has no identified fired-heater inlet
temperature. It changes how a temperature step compares against a feed step, so the
script records the sensitivity of both to it, and `ASSUMPTIONS.md` carries the entry.

Writes `models/economics_reference.json`.

    python -m scripts.analysis.economics_reference
"""
from __future__ import annotations

import json

from scripts import realdata as rd

#: Assumed split of hydrotreater energy between the fired heater and the recycle
#: compressor. The heater dominates in a diesel hydrotreater; both are assumptions.
W_HEAT = 0.7
W_COMPRESSION = 0.3

#: Assumed temperature at which the gas/feed mixture enters the fired heater, after
#: feed/effluent heat exchange. No tag in the package is resolved to it.
HEAT_REFERENCE_C = 300.0

#: Alternatives the sensitivity check reports, to show what the assumption buys.
HEAT_REFERENCE_ALTERNATIVES = (250.0, 275.0, 300.0, 325.0)


def _index(t5: float, gas: float, *, t5_med: float, gas_med: float, ref: float) -> float:
    return (
        W_HEAT * (t5 - ref) / (t5_med - ref)
        + W_COMPRESSION * gas / gas_med
    )


def main() -> None:
    rd.use_utf8_stdout()
    tele = rd.load_telemetry(tuple(rd.LEVERS) + ("F9",))
    clean = tele[rd.clean_operating_mask(tele) & (tele.index <= rd.SPLIT_AT)]

    t5 = clean[rd.tag_id("T5")]
    f2 = clean[rd.tag_id("F2")]
    f26 = clean[rd.tag_id("F26")]
    p13 = clean[rd.tag_id("P13")]
    f9 = clean[rd.tag_id("F9")]

    gas = (f2 / f26) * p13
    density = (f9 / f26).median()

    t5_med = float(t5.median())
    gas_med = float(gas.median())
    f26_med = float(f26.median())

    # What one lever step does to the index at the median regime, for each candidate
    # reference temperature. Only the heat / feed comparison moves with it.
    sensitivity = []
    for ref in HEAT_REFERENCE_ALTERNATIVES:
        base = _index(t5_med, gas_med, t5_med=t5_med, gas_med=gas_med, ref=ref)
        step_t5 = _index(
            t5_med + rd.LEVERS["T5"]["step"], gas_med,
            t5_med=t5_med, gas_med=gas_med, ref=ref,
        )
        feed_up = f26_med + rd.LEVERS["F26"]["step"]
        step_f26 = _index(
            t5_med, float((f2 / feed_up * p13).median()),
            t5_med=t5_med, gas_med=gas_med, ref=ref,
        )
        sensitivity.append({
            "heat_reference_c": ref,
            "energy_pct_per_t5_step": round((step_t5 / base - 1.0) * 100.0, 3),
            "energy_pct_per_f26_step": round((step_f26 / base - 1.0) * 100.0, 3),
        })

    reference = {
        "fitted_on": f"train: date <= {rd.SPLIT_AT.isoformat()}, clean operating rows",
        "n_rows": int(len(clean)),
        "units": "energy index is dimensionless, 1.0 at the median training regime",
        "temperature_tag": rd.tag_id("T5"),
        "feed_tag": rd.tag_id("F26"),
        "gas_tag": rd.tag_id("F2"),
        "pressure_tag": rd.tag_id("P13"),
        "t5_median_c": round(t5_med, 4),
        "gas_term_median": round(gas_med, 6),
        "f26_median_m3_h": round(f26_med, 4),
        "heat_reference_c": HEAT_REFERENCE_C,
        "w_heat": W_HEAT,
        "w_compression": W_COMPRESSION,
        # F9 / F26 = diesel density: the two are one stream in mass and volume
        # (`scripts/realdata.py`). Lets the card state output in t/day as well.
        "density_t_m3": round(float(density), 4),
        "heat_reference_sensitivity": sensitivity,
    }

    out = rd.MODELS_DIR / "economics_reference.json"
    rd.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {rd.rel(out)}")
    print(json.dumps({k: v for k, v in reference.items() if k != "heat_reference_sensitivity"},
                     ensure_ascii=False, indent=2))
    print("\nчувствительность к опорной температуре (одиночный шаг рычага):")
    for row in sensitivity:
        mark = "  <- принято" if row["heat_reference_c"] == HEAT_REFERENCE_C else ""
        print(
            f"  T_ref {row['heat_reference_c']:>5.0f} °C:  T5 +{rd.LEVERS['T5']['step']:g} °C "
            f"-> энергия {row['energy_pct_per_t5_step']:+.2f} %   "
            f"F26 +{rd.LEVERS['F26']['step']:g} м³/ч -> {row['energy_pct_per_f26_step']:+.2f} %{mark}"
        )


if __name__ == "__main__":
    main()
