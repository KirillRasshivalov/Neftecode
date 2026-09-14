from __future__ import annotations

from datetime import datetime

import pandas as pd

from neftecode.domain.state import QualityReading, QualitySource


def compute_age_minutes(state_time: datetime, measured_at: datetime | None) -> float | None:
    if measured_at is None:
        return None
    delta = state_time - measured_at
    return delta.total_seconds() / 60.0


def apply_quality_priority(
    candidates: list[QualityReading],
    priority: list[str] | None = None,
) -> QualityReading | None:
    order = priority or ["lims", "pak", "virtual"]
    by_source = {c.source: c for c in candidates if c.value is not None}
    for source in order:
        reading = by_source.get(source)
        if reading is not None:
            return reading
    return candidates[0] if candidates else None


def ensure_datetime_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series)
