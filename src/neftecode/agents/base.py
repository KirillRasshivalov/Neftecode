from __future__ import annotations

from typing import Protocol

from neftecode.domain.actions import ControlAction
from neftecode.domain.agent_results import QualityAssessment, ReliabilityAssessment
from neftecode.domain.state import ProcessState


class QualityAgent(Protocol):
    def assess(
        self,
        state: ProcessState,
        action: ControlAction | None = None,
    ) -> QualityAssessment: ...


class ReliabilityAgent(Protocol):
    def assess(
        self,
        state: ProcessState,
        action: ControlAction | None = None,
    ) -> ReliabilityAssessment: ...
