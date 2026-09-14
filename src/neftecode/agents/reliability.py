from __future__ import annotations

from neftecode.domain.actions import ControlAction
from neftecode.domain.agent_results import ReliabilityAssessment
from neftecode.domain.state import ProcessState


class ReliabilityAgentStub:
    def assess(
        self,
        state: ProcessState,
        action: ControlAction | None = None,
    ) -> ReliabilityAssessment:
        factors: list[str] = []
        risk = 0.2

        for tag, value in (action.changes if action else state.controllable).items():
            if value is None:
                continue
            if "TEMP" in tag.upper() and (value < 290 or value > 370):
                risk += 0.3
                factors.append(f"{tag} near model boundary ({value})")

        risk = min(1.0, risk)
        allowed = risk < 0.85
        return ReliabilityAssessment(
            risk_index=risk,
            risk_class="high" if risk >= 0.7 else "medium" if risk >= 0.4 else "low",
            risk_factors=factors,
            is_mode_allowed=allowed,
            soft_constraints=[] if allowed else ["reduce severity before optimizing throughput"],
            assumptions=["ReliabilityAgentStub uses proxy thresholds without labeled failures."],
        )


class ReliabilityAgentML:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path

    def assess(
        self,
        state: ProcessState,
        action: ControlAction | None = None,
    ) -> ReliabilityAssessment:
        raise NotImplementedError("ML-2: implement reliability/severity model here")
