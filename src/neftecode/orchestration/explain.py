from __future__ import annotations

from neftecode.domain.agent_results import ScoredScenario
from neftecode.domain.state import ProcessState


def build_explanation(
    state: ProcessState,
    best: ScoredScenario | None,
    *,
    refuse: bool,
    refuse_reason: str | None,
) -> str:
    if refuse:
        return refuse_reason or "Рекомендация отклонена системой безопасности."

    assert best is not None
    sulfur = best.quality.metrics.get("sulfur_mg_kg")
    lines = [
        f"На момент {state.timestamp.isoformat()} выбран вариант '{best.action.label}'.",
        f"Прогноз серы: {sulfur}; риск нарушения спеки: {best.quality.risk_of_spec_breach:.2f}.",
        f"Индекс риска оборудования: {best.reliability.risk_index:.2f} ({best.reliability.risk_class}).",
        "Качество и жёсткие ограничения имеют приоритет над экономическим эффектом.",
    ]
    if best.action.changes:
        changes = ", ".join(f"{k}: {state.controllable.get(k)} → {v}" for k, v in best.action.changes.items())
        lines.insert(1, f"Действие: {changes}.")
    else:
        lines.insert(1, "Действие: режим не менять (noop).")
    return " ".join(lines)
