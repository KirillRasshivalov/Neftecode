"""Real-data preparation shared by the analysis scripts and the bridge.

This is **not** the data layer: `src/neftecode/data/` belongs to Kirill. It exists
so the agents and the orchestrator can run on real data before the real
`ProcessStateBuilder` lands, and so there is a working reference for the rules it
implements. Every rule below comes from the Phase 1 data audit.

Reads only the parquet cache in `data/cache/`, never `data/raw/`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "cache"
MODELS_DIR = REPO_ROOT / "models"

UNIT = "242000"


def tag_id(tag: str, unit: str = UNIT) -> str:
    """Unit-prefixed tag identifier. Bare tag names collide between the two files."""
    return f"{unit}:{tag}"


#: The frozen levers (CLAUDE.md §3). Labels are quoted from the КИП reference;
#: units are inferred because the reference states none; steps are assumptions.
LEVERS: dict[str, dict[str, object]] = {
    "T5": {"step": 2.0, "unit": "°C (inferred)", "label": "Р-201: температура ГСС на выходе"},
    "F26": {"step": 5.0, "unit": "m³/h (inferred)", "label": "Расход сырья на установку (объёмный)"},
    "F2": {"step": 1000.0, "unit": "not stated", "label": "Газовая схема: расход на линии от ЦК-201"},
    "F15": {"step": 50.0, "unit": "not stated", "label": "Расход квенча в Р-202"},
}

#: Chronological split, frozen (CLAUDE.md §6.2).
SPLIT_AT = pd.Timestamp("2025-09-12 14:30")

#: Exact placeholder values on 24-2000 telemetry (audit §2.5).
SENTINELS_242000 = (307.0, 0.0, 10.0)

#: The unit is running when feed flows and the reactor is hot. Shutdowns are
#: written as near-zero floats, which exact sentinel matching does not catch.
RUNNING_MIN = {"F26": 50.0, "T5": 200.0}

#: A row is usable for statistics only when every lever is present and above these.
CLEAN_OPERATING_MIN = {"T5": 200.0, "F26": 50.0, "F2": 10_000.0, "F15": 500.0}

#: Assumed lab reporting delay: a result sampled at t is usable from t + 4 h.
LIMS_DELAY = pd.Timedelta(hours=4)

LIMS_PRODUCT_PARAM = "Mg.Sulfur"   # hydrotreated diesel, мг/кг
LIMS_FEED_PARAM = "Mass.Sulfur"    # hydrotreater feed, % масс.
LIMS_UNIT_MARK = "Гидроочистка"

PAK_SULFUR_SIGNAL = "24-2000:Mg.Sulfur"
#: Analyzer counts as frozen after this many identical consecutive readings (6 h).
PAK_FROZEN_RUN = 36
PAK_MAX_AGE = pd.Timedelta(hours=2)

SPEC_LIMIT_MG_KG = 10.0
#: Lab results above this are screened as possible outliers, not silently dropped.
OUTLIER_ABOVE_MG_KG = 50.0

#: Frozen demo weeks (CLAUDE.md §6.2), all in the held-out region.
DEMO_WEEKS = {
    "stable": pd.Timestamp("2026-01-25 10:00"),
    "risk": pd.Timestamp("2026-07-12 10:00"),
    "degraded": pd.Timestamp("2026-06-21 10:00"),
}


def cache_path(name: str) -> Path:
    path = CACHE_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Put the parquet cache from the data audit into data/cache/."
        )
    return path


def read_cache(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(cache_path(name), columns=columns)
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ImportError("Reading the parquet cache needs pyarrow: pip install pyarrow") from exc


# --------------------------------------------------------------- telemetry


def load_telemetry(tags: tuple[str, ...] = tuple(LEVERS)) -> pd.DataFrame:
    """24-2000 telemetry with sentinels as NaN, indexed by `date`, prefixed columns."""
    df = read_cache("tags_242000", columns=["date", *tags])
    df = df.set_index("date").sort_index()
    df = df.mask(df.isin(SENTINELS_242000))
    df.columns = [tag_id(c) for c in df.columns]
    return df


def running_state(tele: pd.DataFrame) -> pd.Series:
    """True / False where both running tags are present, NaN where either is missing."""
    present = pd.Series(True, index=tele.index)
    running = pd.Series(True, index=tele.index)
    for tag, lo in RUNNING_MIN.items():
        col = tele[tag_id(tag)]
        present &= col.notna()
        running &= col.gt(lo)
    return running.astype("object").where(present, np.nan)


def clean_operating_mask(tele: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=tele.index)
    for tag, lo in CLEAN_OPERATING_MIN.items():
        mask &= tele[tag_id(tag)].gt(lo)
    return mask


def shutdown_ends(tele: pd.DataFrame) -> pd.DatetimeIndex:
    """Timestamps where the unit restarts after running below the feed threshold."""
    feed = tele[tag_id("F26")]
    down = feed.lt(RUNNING_MIN["F26"]).astype(float).where(feed.notna()).ffill()
    restart = (down.shift(1) == 1.0) & (down == 0.0)
    return tele.index[restart.to_numpy()]


# ------------------------------------------------------------------- LIMS


def load_lims() -> pd.DataFrame:
    return read_cache("lims_long")


def load_lab_sulfur(lims: pd.DataFrame | None = None) -> pd.DataFrame:
    """LIMS product sulfur at «Гидроочистка, точка отбора 2», with `available_at`."""
    lims = load_lims() if lims is None else lims
    mask = (
        (lims["param"] == LIMS_PRODUCT_PARAM)
        & lims["point"].str.contains(LIMS_UNIT_MARK)
        & lims["point"].str.contains("'2'", regex=False)
    )
    out = (
        lims.loc[mask, ["timestamp", "value"]]
        .rename(columns={"timestamp": "sampled_at", "value": "sulfur_mg_kg"})
        .sort_values("sampled_at", kind="stable")
        .reset_index(drop=True)
    )
    out["available_at"] = out["sampled_at"] + LIMS_DELAY
    return out


def load_feed_sulfur(lims: pd.DataFrame | None = None) -> pd.DataFrame:
    """LIMS feed sulfur at «Гидроочистка, точка отбора 1», % масс."""
    lims = load_lims() if lims is None else lims
    mask = (
        (lims["param"] == LIMS_FEED_PARAM)
        & lims["point"].str.contains(LIMS_UNIT_MARK)
        & lims["point"].str.contains("'1'", regex=False)
    )
    return (
        lims.loc[mask, ["timestamp", "value"]]
        .rename(columns={"timestamp": "sampled_at", "value": "feed_sulfur_pct"})
        .sort_values("sampled_at", kind="stable")
        .reset_index(drop=True)
    )


# -------------------------------------------------------------------- ПАК


def load_pak_sulfur() -> pd.Series:
    pak = read_cache("pak_long")
    series = pak.loc[pak["signal"] == PAK_SULFUR_SIGNAL].set_index("timestamp")["value"].sort_index()
    return series[~series.index.duplicated()]


# ----------------------------------------------------------------- as-of


def asof_position(index: pd.Index, t: pd.Timestamp) -> int | None:
    """Position of the last index entry at or before `t`, or None."""
    pos = int(index.searchsorted(pd.Timestamp(t), side="right")) - 1
    return pos if pos >= 0 else None


def lab_asof(labels: pd.DataFrame, t: pd.Timestamp) -> pd.Series | None:
    """The newest lab result whose `available_at` is at or before `t`."""
    pos = int(labels["available_at"].searchsorted(pd.Timestamp(t), side="right")) - 1
    return labels.iloc[pos] if pos >= 0 else None


def pak_status(pak: pd.Series, t: pd.Timestamp) -> dict[str, object] | None:
    """Latest analyzer reading at or before `t`, with age and a health verdict."""
    pos = asof_position(pak.index, t)
    if pos is None:
        return None
    measured_at = pak.index[pos]
    age = pd.Timestamp(t) - measured_at
    window = pak.iloc[max(0, pos - PAK_FROZEN_RUN + 1): pos + 1]
    frozen = len(window) >= PAK_FROZEN_RUN and window.nunique() == 1
    return {
        "value": float(pak.iloc[pos]),
        "unit": "ppm",
        "measured_at": measured_at.isoformat(),
        "age_minutes": round(age.total_seconds() / 60.0, 1),
        "frozen": bool(frozen),
        "healthy": bool(age <= PAK_MAX_AGE and not frozen),
    }
