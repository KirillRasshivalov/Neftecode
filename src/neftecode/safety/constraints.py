from __future__ import annotations

from typing import Any

from neftecode.data.config import load_constraints, load_tags_whitelist
from neftecode.domain.actions import ControlAction
from neftecode.domain.agent_results import QualityAssessment, ReliabilityAssessment
from neftecode.domain.state import ProcessState


class HardConstraints:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        whitelist: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or load_constraints()
        self.whitelist = whitelist or load_tags_whitelist()
        self.hard = self.config.get("hard", {})
        self._ranges = {
            item["tag"]: item["range"]
            for item in self.whitelist.get("controllable_parameters", [])
        }

    def check_action(
        self,
        state: ProcessState,
        action: ControlAction,
        quality: QualityAssessment,
        reliability: ReliabilityAssessment,
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []

        sulfur = quality.metrics.get("sulfur_mg_kg")
        sulfur_max = float(self.hard.get("sulfur_mg_kg_max", 10.0))
        if sulfur is not None and sulfur > sulfur_max:
            reasons.append(f"sulfur {sulfur:.3f} > {sulfur_max} mg/kg")

        if not reliability.is_mode_allowed:
            reasons.append("reliability agent marked mode as not allowed")

        for tag, value in action.changes.items():
            rng = self._ranges.get(tag)
            if rng is None:
                reasons.append(f"tag {tag} is not in controllable whitelist")
                continue
            lo, hi = float(rng["min"]), float(rng["max"])
            if value < lo or value > hi:
                marker = " (model assumption)" if rng.get("assumption") else ""
                reasons.append(f"{tag}={value} outside [{lo}, {hi}]{marker}")

        blend_tags = [t for t in {**state.controllable, **action.changes} if "BLEND_RATIO" in t]
        if blend_tags:
            total = 0.0
            for tag in blend_tags:
                total += float(action.changes.get(tag, state.controllable.get(tag, 0.0)))
            target = float(self.hard.get("blend_shares_sum", 1.0))
            tol = float(self.hard.get("blend_shares_tolerance", 1e-3))
            if abs(total - target) > tol:
                reasons.append(f"blend shares sum {total:.4f} != {target}")

        return (len(reasons) == 0, reasons)

    def checked_labels(self) -> list[str]:
        return [
            f"sulfur_mg_kg <= {self.hard.get('sulfur_mg_kg_max', 10.0)}",
            "controllable tags within configured ranges",
            "blend shares sum == 1.0 (when blend tags present)",
            "reliability.is_mode_allowed",
        ]
