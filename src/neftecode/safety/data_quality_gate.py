from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neftecode.data.config import load_constraints
from neftecode.domain.state import ProcessState


@dataclass
class GateResult:
    ok: bool
    reasons: list[str]


class DataQualityGate:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_constraints()
        self.dq = self.config.get("data_quality", {})

    def check(self, state: ProcessState) -> GateResult:
        reasons: list[str] = []

        if state.data_flags.get("complete") is False:
            reasons.append("process state marked incomplete")

        sulfur = state.quality.get("sulfur_mg_kg")
        if self.dq.get("require_quality_source", True) and sulfur is None:
            reasons.append("missing sulfur quality reading")

        if sulfur is not None and sulfur.age_minutes is not None:
            max_age = float(self.dq.get("lims_max_age_minutes", 1440))
            if sulfur.source == "lims" and sulfur.age_minutes > max_age:
                reasons.append(
                    f"LIMS sulfur age {sulfur.age_minutes:.0f} min exceeds {max_age:.0f} min"
                )
            pak_max = float(self.dq.get("pak_max_age_minutes", 360))
            if sulfur.source == "pak" and sulfur.age_minutes > pak_max:
                reasons.append(
                    f"PAK sulfur age {sulfur.age_minutes:.0f} min exceeds {pak_max:.0f} min"
                )

        return GateResult(ok=not reasons, reasons=reasons)
