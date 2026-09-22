"""Real-data preparation shared by the analysis scripts and the bridge.

This is **not** the data layer: `src/neftecode/data/` belongs to Kirill. It exists
so the agents and the orchestrator can run on real data before the real
`ProcessStateBuilder` lands, and so there is a working reference for the rules it
implements. Every rule below comes from the Phase 1 data audit.

Reads only the parquet cache in `data/cache/`, never `data/raw/`.
"""
from __future__ import annotations

import sys
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
#:   is `F14`, not `F15`. It stays unresolved, and a tag whose meaning the reference does
#:   not settle is never a lever.
#: `P13` replaces it: the organisers named reactor inlet pressure as a control variable,
#: and by the physics a higher hydrogen partial pressure reduces sulfur.
LEVERS: dict[str, dict[str, object]] = {
    "T5": {"step": 2.0, "unit": "°C", "label": "Полисеп. Р-201. Температура ГСС на выходе"},
    "F26": {"step": 5.0, "unit": "м³/ч", "label": "Расход гидроочищенного ДТ в цех №8, объёмный"},
    "F2": {"step": 1000.0, "unit": "нм³/ч", "label": "Газовая схема. Расход газа на линии от ЦК-201"},
    "P13": {"step": 0.05, "unit": "МПа", "label": "Полисеп. Р-202. Давление на входе"},
}

#: Chronological split, frozen before any training. No shuffling anywhere.
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

#: Demo weeks, all in the held-out region.
#:
#: `degraded` moved on 19.09.2026. The week first chosen for it turned out to be a
#: shutdown, and a stopped unit is the least interesting refusal there is. From
#: 02.07.2026 the unit runs the whole week while its data fails: the newest lab result
#: is 348 h old and the analyzer is broken for 58 % of the week. The system refuses
#: while it has no quality source, decides on the fallback estimate while only the
#: analyzer is down, and refuses again when fresh catalyst takes the reactor below any
#: temperature the model has seen. The shutdown stays as a scenario of its own.
DEMO_WEEKS = {
    "stable": pd.Timestamp("2026-01-25 10:00"),
    "risk": pd.Timestamp("2026-07-12 10:00"),
    "degraded": pd.Timestamp("2026-07-02 10:00"),
    "shutdown": pd.Timestamp("2026-06-21 10:00"),
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
            f"{path} is missing. Build the cache first: python -m scripts.build_cache "
            "(it needs the organisers' raw files in data/raw/)."
        )
    return path


def cache_columns(name: str) -> list[str]:
    """Column names of a cached table, read from the parquet footer only."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ImportError("Reading the parquet cache needs pyarrow: pip install pyarrow") from exc
    return list(pq.read_schema(cache_path(name)).names)


def read_cache(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(cache_path(name), columns=columns)
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ImportError("Reading the parquet cache needs pyarrow: pip install pyarrow") from exc


# --------------------------------------------------------------- telemetry


def absent_tags(tags: tuple[str, ...]) -> list[str]:
    """The requested tags the telemetry export does not contain at all, prefixed."""
    present = set(cache_columns("tags_242000"))
    return [tag_id(t) for t in tags if t not in present]


def load_telemetry(tags: tuple[str, ...] = tuple(LEVERS), allow_absent: bool = False) -> pd.DataFrame:
    """24-2000 telemetry with sentinels as NaN, indexed by `date`, prefixed columns.

    A tag the export does not contain is an error by default, naming the tag: training on
    an incomplete export must stop, not write NaN into a model. The decision path passes
    `allow_absent=True` instead and gets an all-NaN column, so the state reports the tag
    as missing and the cycle refuses with it named — which is what the task statement
    asks for when data is insufficient.
    """
    missing = absent_tags(tags)
    if missing and not allow_absent:
        raise KeyError(
            f"в выгрузке телеметрии нет тегов: {', '.join(missing)} "
            "(ожидаются по scripts/realdata.py)"
        )
    present = [t for t in tags if tag_id(t) not in missing]
    df = read_cache("tags_242000", columns=["date", *present])
    df = df.set_index("date").sort_index()
    df = df.mask(df.isin(SENTINELS_242000))
    for t in tags:
        if t not in df.columns:
            df[t] = np.nan
    df = df[list(tags)]
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


def load_feature_frame(allow_absent: bool = False) -> pd.DataFrame:
    """Levers, reference-resolved context tags, the gas-to-feed ratio and ПАК sulfur."""
    frame = load_telemetry(tuple(LEVERS) + CONTEXT_TAGS, allow_absent=allow_absent)
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


# --------------------------------------------------------------- catalyst

#: A restart after which the excess reactor temperature falls by this much or more is
#: read as a catalyst change: a catalyst needs a shutdown, and a fresh one runs cooler.
CATALYST_CHANGE_FALL_C = 8.0
#: Days of operation compared on each side of a restart, and the confirmation delay.
CATALYST_WINDOW_DAYS = 21
#: The drift is fitted over the last year of the cycle: long enough to average out the
#: seasonal swing of the feed, short enough to follow the current rate.
CATALYST_DRIFT_DAYS = 365
#: A fresh catalyst loses activity fast at first and only then settles into a steady
#: drift; the first days of a cycle are never extrapolated.
CATALYST_MIN_DAYS = 90


def excess_t5_daily(tele: pd.DataFrame, bands: list[dict]) -> pd.Series:
    """Daily median of how much hotter `242000:T5` runs than usual at the same feed rate.

    "Usual" is the training-period median of T5 in the feed-rate band (the reliability
    reference). Rising excess at constant throughput is the standard signature of a
    catalyst losing activity; it is what the operators compensate by heating.
    """
    ok = clean_operating_mask(tele)
    t5, f26 = tele.loc[ok, tag_id("T5")], tele.loc[ok, tag_id("F26")]
    edges = np.array([b["lo"] for b in bands] + [bands[-1]["hi"]])
    medians = np.array([b["median"] for b in bands])
    idx = np.searchsorted(edges, f26.to_numpy(), side="right") - 1
    inside = (idx >= 0) & (idx < len(bands))
    excess = pd.Series(t5.to_numpy()[inside] - medians[idx[inside]], index=t5.index[inside])
    return excess.resample("D").median().dropna()


#: A stop at least this long may hide a catalyst change; until the change can be
#: confirmed the cycle is reported as not yet established.
LONG_STOP_HOURS = 72.0


def long_restarts(tele: pd.DataFrame, min_hours: float = LONG_STOP_HOURS) -> list[pd.Timestamp]:
    """Restarts that follow a stop of at least `min_hours` — observable when they happen."""
    feed = tele[tag_id("F26")]
    running_at = feed.index[(feed > RUNNING_MIN["F26"]).to_numpy()]
    out = []
    for restart in shutdown_ends(tele):
        before = running_at[running_at < restart]
        if len(before) and (restart - before[-1]) >= pd.Timedelta(hours=min_hours):
            out.append(pd.Timestamp(restart))
    return out


def catalyst_changes(daily: pd.Series, restarts: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Restarts after which the excess temperature falls by `CATALYST_CHANGE_FALL_C`.

    Compares the last `CATALYST_WINDOW_DAYS` days of operation before the stop with the
    first ones after, not a calendar window: a catalyst change is a long shutdown, and
    a calendar window before the restart would hold almost no running days. A burst of
    restarts during one start-up counts once.
    """
    found: list[pd.Timestamp] = []
    for restart in restarts:
        if found and restart - found[-1] < pd.Timedelta(days=CATALYST_WINDOW_DAYS):
            continue
        before = daily.loc[: restart - pd.Timedelta(seconds=1)].tail(CATALYST_WINDOW_DAYS)
        after = daily.loc[restart:].head(CATALYST_WINDOW_DAYS)
        if len(before) < 7 or len(after) < 7:
            continue
        if before.median() - after.median() >= CATALYST_CHANGE_FALL_C:
            found.append(pd.Timestamp(restart))
    return found


def catalyst_status(
    daily: pd.Series,
    changes: list[pd.Timestamp],
    t: pd.Timestamp,
    eor_excess_c: float,
    long_stops: list[pd.Timestamp] | None = None,
) -> dict[str, object]:
    """Where the catalyst is in its cycle as of `t`, from data available at `t` only.

    Uses whole days before `t`. That a restart was a catalyst change is known only once
    `CATALYST_WINDOW_DAYS` days after it have been seen — that is how it is detected —
    so for that long after any long stop the cycle is reported as not yet established.
    Which stops were changes is never read from the future.
    """
    t = pd.Timestamp(t)
    known = daily.loc[: t.normalize() - pd.Timedelta(seconds=1)]
    window = pd.Timedelta(days=CATALYST_WINDOW_DAYS)
    confirmed = [c for c in changes if c + window <= t]
    unsettled = [r for r in (long_stops or []) if r <= t < r + window]
    out: dict[str, object] = {"eor_excess_c": round(float(eor_excess_c), 2)}
    if unsettled:
        out.update(state="start_up", restart=unsettled[-1].isoformat())
        return out
    start = confirmed[-1] if confirmed else known.index.min()
    out["cycle_start"] = pd.Timestamp(start).isoformat()
    out["days_in_cycle"] = int((t - pd.Timestamp(start)).days)
    if confirmed:
        # skip the start-of-run transient of a catalyst we saw being loaded
        steady = known.loc[pd.Timestamp(start) + pd.Timedelta(days=CATALYST_MIN_DAYS):]
    else:
        steady = known.loc[start:]
    if out["days_in_cycle"] < CATALYST_MIN_DAYS or len(steady) < CATALYST_MIN_DAYS // 3:
        out["state"] = "too_short"
        return out
    recent_days = steady.tail(CATALYST_DRIFT_DAYS)
    x = (recent_days.index - recent_days.index[0]).days.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, recent_days.to_numpy(dtype=float), 1)
    # Wear is estimated on the pessimistic side: the trend line is steady but misses the
    # speed-up near the end of a run, the last fortnight is noisy but catches it. A warning
    # that comes late is worse than one that comes early, so the older of the two counts.
    trend = float(intercept + slope * x[-1])
    recent = float(known.tail(14).median())
    now = max(trend, recent)
    slope = float(slope)
    out.update(
        excess_now_c=round(now, 2),
        excess_trend_c=round(trend, 2),
        excess_recent_c=round(recent, 2),
        drift_c_per_month=round(slope * 30.0, 3),
    )
    if now >= eor_excess_c:
        out.update(state="eor_reached", days_to_eor=0)
    elif slope <= 0.0:
        out.update(state="no_drift", days_to_eor=None)
    else:
        out.update(state="ageing", days_to_eor=int(round((eor_excess_c - now) / slope)))
    return out


def use_utf8_stdout() -> None:
    """Output carries box rules, arrows and units; a Windows console defaults to cp1251."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass
