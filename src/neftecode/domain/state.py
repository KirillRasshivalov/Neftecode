from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


QualitySource = Literal["lims", "pak", "virtual", "unknown"]


class TagValue(BaseModel):
    tag: str
    value: float | None
    unit: str | None = None
    source: str | None = None


class QualityReading(BaseModel):
    metric: str
    value: float | None
    unit: str
    source: QualitySource
    measured_at: datetime | None = None
    age_minutes: float | None = None


class ProcessState(BaseModel):
    timestamp: datetime
    kip: dict[str, TagValue] = Field(default_factory=dict)
    quality: dict[str, QualityReading] = Field(default_factory=dict)
    controllable: dict[str, float] = Field(default_factory=dict)
    feature_windows: dict[str, float | None] | None = None
    data_flags: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def windows_for_agents(self) -> dict[str, float | None] | None:
        if self.feature_windows is not None:
            return self.feature_windows
        raw = self.data_flags.get("feature_windows")
        return raw if isinstance(raw, dict) else None
