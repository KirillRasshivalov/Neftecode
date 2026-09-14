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
    data_flags: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
