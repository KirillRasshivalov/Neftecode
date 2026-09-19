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


#: The levers. Labels and units are quoted from the **corrected** 24-2000 reference the
#: organisers issued on 16.09.2026 (`теги АВТ_24-2000.xlsx`), which for the first time
#: states a physical quantity and a unit per tag. Steps remain assumptions.
#:
#: Two corrections against our earlier set, both checked against the data:
#: - `F26` is the hydrotreated diesel leaving for shop 8, not the feed. It stays the
#:   throughput lever anyway: `F9` (feed, mass) divided by `F26` is 0.850 t/m³, exactly
#:   diesel density, and the two correlate at 0.9999 — one stream, two units.
#: - `F15` is dropped. The reference calls it the volumetric feed, but 3 400 m³/h against
#:   a mass feed of 220 t/h implies a density of 0.065, so it is not that; and the quench
#:   is `F14`, not `F15`. It stays unresolved and may not be a lever (CLAUDE.md §6.1).
#: `P13` replaces it: the organisers named reactor inlet pressure as a control variable,
#: and CLAUDE.md §11 requires higher hydrogen partial pressure to reduce sulfur.
LEVERS: dict[str, dict[str, object]] = {
    "T5": {"step": 2.0, "unit": "°C", "label": "Полисеп. Р-201. Температура ГСС на выходе"},
    "F26": {"step": 5.0, "unit": "м³/ч", "label": "Расход гидроочищенного ДТ в цех №8, объёмный"},
    "F2": {"step": 1000.0, "unit": "нм³/ч", "label": "Газовая схема. Расход газа на линии от ЦК-201"},
    "P13": {"step": 0.05, "unit": "МПа", "label": "Полисеп. Р-202. Давление на входе"},
}

#: Chronological split, frozen (CLAUDE.md §6.2).
SPLIT_AT = pd.Timestamp("2025-09-12 14:30")

#: Exact placeholder values on 24-2000 telemetry (audit §2.5).
SENTINELS_242000 = (307.0, 0.0, 10.0)

#: The unit is running when feed flows and the reactor is hot. Shutdowns are
#: written as near-zero floats, which exact sentinel matching does not catch.
RUNNING_MIN = {"F26": 50.0, "T5": 200.0}

#: A row is usable for statistics only when every lever is present and above these.
CLEAN_OPERATING_MIN = {"T5": 200.0, "F26": 50.0, "F2": 10_000.0, "P13": 1.0}

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


def rel(path: Path) -> str:
    """A path relative to the repository root, for logs that end up in shared artifacts."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


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


# --------------------------------------------------------------- features

#: Non-lever tags used as features, all with a stated quantity in the corrected
#: reference. `Q21` is the product sulfur analyzer inside the telemetry file: its median
#: is 8.43 ppm against the ПАК export's 8.45, so it is the same measurement on the
#: 10-minute grid. `T6` is the second reactor's inlet temperature.
CONTEXT_TAGS = ("F1", "P3", "W4", "F9", "T16", "F17", "T6", "Q21")

#: Quantity and unit per tag, quoted from the corrected reference (16.09.2026).
TAG_UNITS = {
    "F1": "м³/ч", "F2": "нм³/ч", "P3": "МПа", "W4": "т/ч", "T5": "°C", "T6": "°C",
    "W7": "т/ч", "P8": "МПа", "F9": "т/ч", "W10": "т/ч", "T11": "°C", "T12": "°C",
    "P13": "МПа", "F14": "т/ч", "F15": "не разрешён", "T16": "°C", "F17": "т/ч",
    "T18": "°C", "F19": "т/ч", "Q20": "ppm", "Q21": "ppm", "F22": "нм³/ч",
    "T23": "°C", "P24": "МПа", "F25": "нм³/ч", "F26": "м³/ч",
}

#: Every feature is a trailing mean over these windows, in hours.
FEATURE_WINDOWS_H = (2, 6, 24)
ROWS_PER_HOUR = 6

#: Gas-to-feed ratio, standing in for the hydrogen-to-feed ratio the package lacks.
RATIO_ID = f"{tag_id('F2')}/{tag_id('F26')}"
PAK_FEATURE = "pak:sulfur"

#: Lab-history features. Continuous ones come from the non-outlier series the quality
#: agent smooths; the binary ones count every running result, outliers included.
LAB_FEATURE_NAMES = (
    "lab_ewma",
    "lab_last",
    "lab_age_h",
    "lab_above_last",
    "lab_above_share7",
    "lab_hours_since_above",
)
LAB_HISTORY_WINDOW = 7
HOURS_SINCE_ABOVE_CAP = 720.0


def load_feature_frame() -> pd.DataFrame:
    """Levers, reference-resolved context tags, the gas-to-feed ratio and ПАК sulfur."""
    frame = load_telemetry(tuple(LEVERS) + CONTEXT_TAGS)
    feed = frame[tag_id("F26")]
    frame[RATIO_ID] = (frame[tag_id("F2")] / feed).where(feed > RUNNING_MIN["F26"])
    frame[PAK_FEATURE] = load_pak_sulfur().reindex(frame.index)
    return frame


def window_means(frame: pd.DataFrame) -> pd.DataFrame:
    """Trailing means named `<column>|<hours>h`, requiring 80 % of the rows present."""
    out = {}
    for column in frame.columns:
        series = frame[column]
        for hours in FEATURE_WINDOWS_H:
            n = hours * ROWS_PER_HOUR
            out[f"{column}|{hours}h"] = series.rolling(n, min_periods=int(0.8 * n)).mean()
    return pd.DataFrame(out, index=frame.index)


def asof_matrix(features: pd.DataFrame, times: pd.DatetimeIndex, lag_hours: int = 0) -> np.ndarray:
    """Feature rows as of `times - lag_hours`; NaN where nothing was available yet."""
    pos = features.index.searchsorted(times - pd.Timedelta(hours=lag_hours), side="right") - 1
    matrix = features.to_numpy()[np.clip(pos, 0, None)].astype(float)
    matrix[pos < 0] = np.nan
    return matrix


def lab_history(
    running: pd.DataFrame,
    usable: pd.DataFrame,
    ewma: np.ndarray,
    cutoffs: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Lab-history features from results already usable at each cutoff.

    `running` is every lab result taken while the unit ran, outliers included, and
    `usable` is the non-outlier subset the quality agent smooths, with `ewma` the
    smoothed level right after each of its results. Nothing here reads a result whose
    `available_at` is later than the cutoff, so the as-of guarantee holds.
    """
    cut = pd.DatetimeIndex(cutoffs).to_numpy()
    hour = np.timedelta64(1, "h")

    u_pos = np.searchsorted(usable["available_at"].to_numpy(), cut, side="right") - 1
    u_sampled = usable["sampled_at"].to_numpy()
    u_value = usable["sulfur_mg_kg"].to_numpy()
    safe = np.clip(u_pos, 0, None)
    seen = u_pos >= 0

    a_pos = np.searchsorted(running["available_at"].to_numpy(), cut, side="right") - 1
    a_sampled = running["sampled_at"].to_numpy()
    a_above = running["above_spec"].to_numpy().astype(float)
    above_at = np.flatnonzero(a_above > 0)

    share = np.full(len(cut), np.nan)
    since = np.full(len(cut), HOURS_SINCE_ABOVE_CAP)
    for i, k in enumerate(a_pos):
        if k < 0:
            continue
        share[i] = a_above[max(0, k - LAB_HISTORY_WINDOW + 1) : k + 1].mean()
        j = int(np.searchsorted(above_at, k, side="right")) - 1
        if j >= 0:
            elapsed = float((cut[i] - a_sampled[above_at[j]]) / hour)
            since[i] = min(elapsed, HOURS_SINCE_ABOVE_CAP)

    return pd.DataFrame({
        "lab_ewma": np.where(seen, ewma[safe], np.nan),
        "lab_last": np.where(seen, u_value[safe], np.nan),
        "lab_age_h": np.where(seen, (cut - u_sampled[safe]) / hour, np.nan),
        "lab_above_last": np.where(a_pos >= 0, a_above[np.clip(a_pos, 0, None)], np.nan),
        "lab_above_share7": share,
        "lab_hours_since_above": since,
    })
