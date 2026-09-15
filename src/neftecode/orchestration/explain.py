from __future__ import annotations

from neftecode.domain.agent_results import ScoredScenario
from neftecode.domain.state import ProcessState

SCOPE_NOTE = (
    "Ограничение по сере проверяется в гидроочищенном ДТ, а не в товарном "
    "(допущение: данных блендинга в пакете нет)."
)
PRIORITY_NOTE = "Качество и жёсткие ограничения имеют приоритет над экономическим эффектом."


def fmt(value: float | None) -> str:
    """Operator-readable number: no raw floats, spaces as thousands separators."""
    if value is None:
        return "—"
    v = float(value)
    if abs(v) >= 1000:
        return f"{v:,.0f}".replace(",", " ")
    if abs(v) >= 100:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _unit(state: ProcessState, tag: str) -> str:
    info = state.kip.get(tag)
    unit = (info.unit or "") if info else ""
    unit = unit.split(" (")[0].strip()
    return "" if unit in ("", "not stated") else unit


def _interval(scenario: ScoredScenario | None) -> dict:
    if scenario is None:
        return {}
    return (scenario.quality.details or {}).get("intervals", {}).get("sulfur_mg_kg", {})


def describe_action(state: ProcessState, scenario: ScoredScenario) -> str:
    parts = []
    for tag, new in scenario.action.changes.items():
        current = state.controllable.get(tag)
        if current is None or abs(float(new) - float(current)) < 1e-9:
            continue
        unit = _unit(state, tag)
        parts.append(f"{tag}: {fmt(current)} → {fmt(new)}{' ' + unit if unit else ''}")
    return ", ".join(parts) or "режим не менять"


def build_explanation(
    state: ProcessState,
    best: ScoredScenario | None,
    *,
    refuse: bool,
    refuse_reason: str | None,
    outcome: str | None = None,
    why: str | None = None,
    triggers: list[str] | None = None,
    hold: ScoredScenario | None = None,
) -> str:
    if refuse:
        return refuse_reason or "Рекомендация отклонена системой безопасности."

    assert best is not None
    lab = state.quality.get("sulfur_mg_kg")
    if lab is not None and lab.value is not None and lab.age_minutes is not None:
        lab_text = f"Последний анализ ЛИМС: {lab.value:.1f} мг/кг, {lab.age_minutes / 60:.0f} ч назад."
    else:
        lab_text = "Лабораторного результата нет."

    after = _interval(best)
    after_mean = after.get("mean", best.quality.metrics.get("sulfur_mg_kg"))
    is_hold = outcome == "hold" or best.action.is_noop()

    if is_hold:
        lines = [
            f"Режим не менять: {why}." if why else "Режим не менять.",
            f"Признаки проблемы: {'; '.join(triggers)}." if triggers else "",
            lab_text,
            f"Прогноз серы {fmt(after_mean)} мг/кг, p95 {fmt(after.get('p95'))} мг/кг.",
        ]
    else:
        before = _interval(hold)
        before_mean = before.get("mean", hold.quality.metrics.get("sulfur_mg_kg") if hold else None)
        lines = [
            f"Рекомендация: {describe_action(state, best)}.",
            f"Причина: {'; '.join(triggers)}." if triggers else "",
            lab_text,
            (
                f"Ожидаемый эффект: прогноз серы {fmt(before_mean)} → {fmt(after_mean)} мг/кг, "
                f"p95 {fmt(before.get('p95'))} → {fmt(after.get('p95'))} мг/кг; "
                f"тяжесть режима {best.reliability.risk_class} ({best.reliability.risk_index:.2f})."
            ),
            f"Почему этот вариант: {why}." if why else "",
        ]
    lines += [SCOPE_NOTE, PRIORITY_NOTE]
    return " ".join(line for line in lines if line)
