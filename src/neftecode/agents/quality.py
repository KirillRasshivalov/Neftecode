from __future__ import annotations

from neftecode.domain.actions import ControlAction
from neftecode.domain.agent_results import QualityAssessment
from neftecode.domain.state import ProcessState


class QualityAgentStub:
    def assess(
        self,
        state: ProcessState,
        action: ControlAction | None = None,
    ) -> QualityAssessment:
        sulfur_reading = state.quality.get("sulfur_mg_kg")
        sulfur = sulfur_reading.value if sulfur_reading else None

        if sulfur is not None and action and action.changes:
            delta_t = action.changes.get("PLACEHOLDER_HDT_TEMP")
            if delta_t is not None:
                current = state.controllable.get("PLACEHOLDER_HDT_TEMP")
                if current is not None:
                    sulfur = max(0.0, sulfur - 0.05 * (delta_t - current))

        risk = 0.0
        if sulfur is not None:
            risk = max(0.0, min(1.0, (sulfur - 8.0) / 4.0))

        return QualityAssessment(
            metrics={"sulfur_mg_kg": sulfur},
            risk_of_spec_breach=risk,
            confidence=0.4 if state.data_flags.get("scaffold_mode") else 0.6,
            horizon_minutes=60,
            features_used=sorted((action.changes if action else {}).keys()),
            assumptions=["QualityAgentStub uses heuristic sulfur response, not a trained VAK."],
        )


class QualityAgentML:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self._model = None

    def assess(
        self,
        state: ProcessState,
        action: ControlAction | None = None,
    ) -> QualityAssessment:
        raise NotImplementedError("ML-1: implement model load + feature mapping here")
