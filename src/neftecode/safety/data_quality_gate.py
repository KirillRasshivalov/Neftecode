from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from neftecode.data.config import load_constraints
from neftecode.domain.state import ProcessState


@dataclass
class GateResult:
    ok: bool
    reasons: list[str]
    warnings: list[str] = field(default_factory=list)


def _hours(minutes: float) -> str:
    hours = minutes / 60.0
    if abs(hours - round(hours)) < 1e-6:
        return f"{hours:.0f}"
    return f"{hours:.1f}"


class DataQualityGate:
    """Отказ только если нет ни свежего ЛИМС, ни исправного ПАК."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_constraints()
        self.dq = self.config.get("data_quality", {})

    def check(self, state: ProcessState) -> GateResult:
        reasons: list[str] = []
        warnings: list[str] = []

        if state.data_flags.get("complete") is False:
            reasons.append("состояние процесса неполное")

        sulfur = state.quality.get("sulfur_mg_kg")
        pak = state.data_flags.get("pak_sulfur") or {}
        pak_healthy = pak.get("healthy") is True
        lims_max = float(self.dq.get("lims_max_age_minutes", 1800))
        pak_max = float(self.dq.get("pak_max_age_minutes", 360))
        require = bool(self.dq.get("require_quality_source", True))

        if sulfur is None:
            if require and not pak_healthy:
                reasons.append(
                    "нет показания серы: анализ ЛИМС отсутствует, "
                    "поточный анализатор недоступен или неисправен"
                )
            elif require and pak_healthy:
                warnings.append(
                    "анализ ЛИМС отсутствует — опираемся на исправный поточный анализатор"
                )
            return GateResult(ok=not reasons, reasons=reasons, warnings=warnings)

        age = sulfur.age_minutes
        if sulfur.source == "lims" and age is not None and age > lims_max:
            if pak_healthy:
                warnings.append(
                    f"анализ ЛИМС {_hours(age)} ч назад при допустимых {_hours(lims_max)} ч — "
                    "опираемся на исправный поточный анализатор"
                )
            else:
                reasons.append(
                    f"анализ ЛИМС {_hours(age)} ч назад при допустимых {_hours(lims_max)} ч; "
                    "поточный анализатор недоступен или неисправен"
                )

        if sulfur.source == "pak" and age is not None and age > pak_max:
            if pak_healthy:
                warnings.append(
                    f"показание ПАК в карточке {_hours(age)} ч назад при пороге "
                    f"{_hours(pak_max)} ч — статус анализатора: исправен"
                )
            else:
                reasons.append(
                    f"показание ПАК {_hours(age)} ч назад при допустимых {_hours(pak_max)} ч"
                )

        return GateResult(ok=not reasons, reasons=reasons, warnings=warnings)
