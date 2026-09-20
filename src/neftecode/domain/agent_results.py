from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from neftecode.domain.actions import ControlAction


class MetricInterval(BaseModel):
    mean: float | None = None
    p05: float | None = None
    p95: float | None = None


class QualityAssessment(BaseModel):
    metrics: dict[str, float | None] = Field(default_factory=dict)
    risk_of_spec_breach: float = 0.0
    confidence: float = 0.0
    horizon_minutes: int = 0
    features_used: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    intervals: dict[str, MetricInterval] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)

    def interval_for(self, metric: str) -> MetricInterval | None:
        if metric in self.intervals:
            return self.intervals[metric]
        raw = (self.details or {}).get("intervals", {}).get(metric)
        if isinstance(raw, dict):
            return MetricInterval.model_validate(raw)
        return None


class ReliabilityAssessment(BaseModel):
    risk_index: float = 0.0
    risk_class: str = "unknown"
    risk_factors: list[str] = Field(default_factory=list)
    is_mode_allowed: bool = True
    soft_constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class ScoredScenario(BaseModel):
    action: ControlAction
    quality: QualityAssessment
    reliability: ReliabilityAssessment
    feasible: bool = True
    rejection_reasons: list[str] = Field(default_factory=list)
    score: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    constraint_margins: dict[str, float] = Field(default_factory=dict)
