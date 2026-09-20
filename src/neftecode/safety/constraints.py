from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from neftecode.data.config import load_constraints, load_tags_whitelist
from neftecode.domain.actions import ControlAction
from neftecode.domain.agent_results import QualityAssessment, ReliabilityAssessment
from neftecode.domain.state import ProcessState


@dataclass
class ConstraintResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    margins: dict[str, float] = field(default_factory=dict)

    def __iter__(self):
        return iter((self.ok, self.reasons))


class HardConstraints:
    """Жёсткие проверки. Сера — по p95 интервала, иначе по среднему."""

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
    ) -> ConstraintResult:
        reasons: list[str] = []
        margins: dict[str, float] = {}

        sulfur_max = float(self.hard.get("sulfur_mg_kg_max", 10.0))
        sulfur_value, sulfur_label = self._sulfur_for_check(quality)
        if sulfur_value is not None:
            margins["sulfur_mg_kg"] = round(sulfur_max - float(sulfur_value), 4)
            if sulfur_value > sulfur_max:
                reasons.append(
                    f"сера {sulfur_label} {sulfur_value:.3f} > {sulfur_max:g} мг/кг"
                )

        if not reliability.is_mode_allowed:
            detail = [
                f
                for f in reliability.risk_factors
                if any(token in f for token in ("диапазон", "Индекс тяжести", "не работает", "Катализатор"))
            ]
            if detail:
                reasons.extend(detail)
            else:
                reasons.append("режим не допускается агентом надёжности")

        for tag, value in action.changes.items():
            rng = self._ranges.get(tag)
            if rng is None:
                reasons.append(f"тег {tag} отсутствует в белом списке управляемых")
                continue
            lo, hi = float(rng["min"]), float(rng["max"])
            if value < lo or value > hi:
                marker = " (модельное допущение)" if rng.get("assumption") else ""
                reasons.append(f"{tag}={value} вне [{lo}, {hi}]{marker}")
                margins[f"range:{tag}"] = round(min(value - lo, hi - value), 4)

        blend_tags = [t for t in {**state.controllable, **action.changes} if "BLEND_RATIO" in t]
        if blend_tags:
            total = 0.0
            for tag in blend_tags:
                total += float(action.changes.get(tag, state.controllable.get(tag, 0.0)))
            target = float(self.hard.get("blend_shares_sum", 1.0))
            tol = float(self.hard.get("blend_shares_tolerance", 1e-3))
            margins["blend_shares_sum"] = round(target - total, 4)
            if abs(total - target) > tol:
                reasons.append(f"сумма долей блендинга {total:.4f} != {target}")

        return ConstraintResult(ok=len(reasons) == 0, reasons=reasons, margins=margins)

    @staticmethod
    def _sulfur_for_check(quality: QualityAssessment) -> tuple[float | None, str]:
        interval = quality.interval_for("sulfur_mg_kg")
        if interval is not None and interval.p95 is not None:
            return float(interval.p95), "p95"
        mean = quality.metrics.get("sulfur_mg_kg")
        if mean is not None:
            return float(mean), "среднее"
        return None, "сера"

    def checked_labels(self) -> list[str]:
        return [
            f"sulfur_mg_kg <= {self.hard.get('sulfur_mg_kg_max', 10.0)} (по p95, иначе среднее)",
            "controllable tags within configured ranges",
            "blend shares sum == 1.0 (when blend tags present)",
            "reliability.is_mode_allowed",
        ]
