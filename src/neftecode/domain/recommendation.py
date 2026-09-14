from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from neftecode.domain.actions import ControlAction
from neftecode.domain.agent_results import ScoredScenario


class OperatorRecommendation(BaseModel):
    timestamp: datetime
    refuse: bool = False
    refuse_reason: str | None = None

    problem_or_risk: str | None = None
    proposed_action: ControlAction | None = None
    expected_effect: dict[str, Any] = Field(default_factory=dict)
    constraints_checked: list[str] = Field(default_factory=list)
    confidence: float | None = None
    explanation: str | None = None

    alternatives: list[ScoredScenario] = Field(default_factory=list)
    audit: dict[str, Any] = Field(default_factory=dict)
