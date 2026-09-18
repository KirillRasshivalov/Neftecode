"""Blend component properties: measured where the lab measured them, derived otherwise.

The organisers' blending scheme has three components — hydrotreated diesel, kerosene
and gas oil — each described by density, sulfur, cetane number and T95. Only the
first is sampled at its own point. The other two are proxied by atmospheric-unit cuts
that the lab does sample:

- kerosene  → «АВТ, точка отбора 2»: T95 296 °C, cloud point −23 °C, the light cut;
- gas oil   → «АВТ, точка отбора 1»: D15 882 kg/m³, the heavy cut.

Those cuts are straight-run, and straight-run material cannot go into a product held
to 10 mg/kg of sulfur. They are therefore taken as their **hydrotreated** grades: the
change hydrotreating makes to the diesel cut (feed «Гидроочистка т.о. 1» against
product «т.о. 2») is applied to their density and distillation.

Cetane is not sampled on either cut. It is computed with the two-variable cetane index
of ASTM D976 from density and the 50 % distillation point, after checking that the
same formula reproduces the cetane numbers the lab did measure on the hydrotreated
diesel. Sulfur of the two proxies stays an assumption, set in `configs/blending.yaml`.

Writes `models/blend_components.json`.

    python -m scripts.analysis.blend_components
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from scripts import realdata as rd

HT_PRODUCT = ("Гидроочистка", "'2'")
HT_FEED = ("Гидроочистка", "'1'")
KEROSENE_CUT = ("АВТ", "'2'")
GAS_OIL_CUT = ("АВТ", "'1'")

PROPERTIES = {"D15": "density_kg_m3", "50%.T": "t50_c", "95%.T": "t95_c"}

#: Physically possible for a diesel-range cut. The lab file holds a handful of zeros
#: and similar entry errors; they are dropped, counted and reported, never imputed.
PLAUSIBLE = {"D15": (700.0, 1000.0), "50%.T": (100.0, 450.0), "95%.T": (150.0, 500.0),
             "CetaneNumber": (20.0, 80.0)}


def cetane_index_d976(density_kg_m3: float, t50_c: float) -> float:
    """Two-variable cetane index, ASTM D976: density at 15 °C and mid-boiling point."""
    d = density_kg_m3 / 1000.0
    b = t50_c
    return 454.74 - 1641.416 * d + 774.74 * d * d - 0.554 * b + 97.803 * math.log10(b) ** 2


def at_point(lims: pd.DataFrame, unit: str, point: str) -> pd.DataFrame:
    mask = lims["point"].str.contains(unit, regex=False) & lims["point"].str.contains(point, regex=False)
    frame = lims[mask]
    keep = pd.Series(True, index=frame.index)
    for param, (lo, hi) in PLAUSIBLE.items():
        rows = frame["param"] == param
        keep &= ~rows | frame["value"].between(lo, hi)
    return frame[keep]


def implausible(lims: pd.DataFrame) -> dict[str, int]:
    """How many values of each checked parameter fall outside the plausible range."""
    return {
        param: int((~lims.loc[lims["param"] == param, "value"].between(lo, hi)).sum())
        for param, (lo, hi) in PLAUSIBLE.items()
    }


def medians(frame: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for param, name in PROPERTIES.items():
        values = frame.loc[frame["param"] == param, "value"].dropna()
        out[name] = {"value": round(float(values.median()), 2), "n": int(len(values))}
    return out


def validate_d976(product: pd.DataFrame) -> dict:
    """Cetane index against measured cetane, sample by sample, on the hydrotreated diesel."""
    wide = product.pivot_table(index="timestamp", columns="param", values="value", aggfunc="first")
    need = ["CetaneNumber", "D15", "50%.T"]
    paired = wide.dropna(subset=[c for c in need if c in wide.columns])
    if not all(c in paired.columns for c in need) or paired.empty:
        return {"n_pairs": 0}
    computed = np.array([cetane_index_d976(d, t) for d, t in zip(paired["D15"], paired["50%.T"])])
    measured = paired["CetaneNumber"].to_numpy()
    error = computed - measured
    return {
        "n_pairs": int(len(paired)),
        "bias": round(float(error.mean()), 3),
        "mae": round(float(np.abs(error).mean()), 3),
        "max_abs_error": round(float(np.abs(error).max()), 3),
        "measured_median": round(float(np.median(measured)), 2),
        "computed_median": round(float(np.median(computed)), 2),
    }


def main() -> None:
    lims = rd.load_lims()
    product = at_point(lims, *HT_PRODUCT)
    feed = at_point(lims, *HT_FEED)

    diesel = medians(product)
    diesel["sulfur_mg_kg"] = {
        "value": round(float(product.loc[product["param"] == "Mg.Sulfur", "value"].median()), 2),
        "n": int((product["param"] == "Mg.Sulfur").sum()),
    }
    cetane = product.loc[product["param"] == "CetaneNumber", "value"].dropna()
    diesel["cetane"] = {"value": round(float(cetane.median()), 2), "n": int(len(cetane))}

    # What hydrotreating does to a cut's density and distillation, measured on diesel.
    feed_props = medians(feed)
    shift = {name: round(diesel[name]["value"] - feed_props[name]["value"], 2) for name in PROPERTIES.values()}

    validation = validate_d976(product)
    components = {
        "hydrotreated_diesel": {
            "label": "Очищенный дизель (гидроочистка, т.о. 2)",
            "properties": {k: v["value"] for k, v in diesel.items() if k != "t50_c"},
            "n": {k: v["n"] for k, v in diesel.items()},
            "sources": {
                "density_kg_m3": "lims: Гидроочистка т.о. 2, D15",
                "t95_c": "lims: Гидроочистка т.о. 2, 95%.T",
                "sulfur_mg_kg": "lims: Гидроочистка т.о. 2, Mg.Sulfur",
                "cetane": "lims: Гидроочистка т.о. 2, CetaneNumber",
            },
        }
    }
    for key, label, cut in (
        ("kerosene", "Керосин (прокси: АВТ т.о. 2, гидроочищенный)", KEROSENE_CUT),
        ("gas_oil", "Газойль (прокси: АВТ т.о. 1, гидроочищенный)", GAS_OIL_CUT),
    ):
        raw = medians(at_point(lims, *cut))
        treated = {name: round(raw[name]["value"] + shift[name], 2) for name in PROPERTIES.values()}
        components[key] = {
            "label": label,
            "properties": {
                "density_kg_m3": treated["density_kg_m3"],
                "t95_c": treated["t95_c"],
                "cetane": round(cetane_index_d976(treated["density_kg_m3"], treated["t50_c"]), 2),
            },
            "straight_run": {name: raw[name]["value"] for name in PROPERTIES.values()},
            "n": {name: raw[name]["n"] for name in PROPERTIES.values()},
            "sources": {
                "density_kg_m3": f"lims: АВТ т.о. {cut[1].strip(chr(39))}, D15 + сдвиг гидроочистки",
                "t95_c": f"lims: АВТ т.о. {cut[1].strip(chr(39))}, 95%.T + сдвиг гидроочистки",
                "cetane": "ASTM D976 по плотности и T50, формула проверена на гидроочищенном дизеле",
                "sulfur_mg_kg": "допущение, configs/blending.yaml",
            },
        }

    report = {
        "implausible_values_dropped": implausible(lims),
        "hydrotreating_shift": shift,
        "hydrotreating_shift_note": "продукт минус сырьё на гидроочистке (т.о. 2 минус т.о. 1), медианы",
        "cetane_index_d976_validation": validation,
        "cetane_index_of_diesel_median": round(
            cetane_index_d976(diesel["density_kg_m3"]["value"], diesel["t50_c"]["value"]), 2
        ),
        "components": components,
    }
    out = rd.MODELS_DIR / "blend_components.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {out}\n")
    print(f"implausible lab values dropped: {report['implausible_values_dropped']}")
    print(f"hydrotreating shift (product - feed): {shift}")
    print(f"ASTM D976 on hydrotreated diesel: {validation}")
    print(f"  cetane index at the median properties: {report['cetane_index_of_diesel_median']} "
          f"vs measured median {diesel['cetane']['value']}\n")
    for key, comp in components.items():
        print(f"{key:<20} {comp['properties']}")


if __name__ == "__main__":
    main()
