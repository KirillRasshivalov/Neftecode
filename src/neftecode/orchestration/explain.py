from __future__ import annotations

from neftecode.domain.agent_results import ScoredScenario
from neftecode.domain.state import ProcessState

SCOPE_NOTE = (
    "Режим гидроочистки проверяется по сере гидроочищенного ДТ; норма 10 мг/кг по ТЗ "
    "относится к товарной смеси и проверяется на ней."
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


#: How the quality agent arrived at the probability, in operator language.
RISK_SOURCES = {
    "classifier": "классификатор по поточному анализатору",
    "interval": "оценка по интервалу прогноза",
}


def _risk(scenario: ScoredScenario) -> str:
    source = (scenario.quality.details or {}).get("risk_source")
    return (
        f"Вероятность нарушения {scenario.quality.risk_of_spec_breach:.0%} "
        f"({RISK_SOURCES.get(source, 'оценка')})."
    )


#: Blend components in operator language.
COMPONENT_NAMES = {"hydrotreated_diesel": "очищенный ДТ", "kerosene": "керосин", "gas_oil": "газойль"}


def describe_blend(blend: dict | None) -> str:
    """One sentence on the commercial blend, or on why there is none."""
    if not blend:
        return ""
    if blend.get("outcome") != "blend" or not blend.get("best"):
        return f"Товарная смесь: {blend.get('refuse_reason') or 'допустимой нет'}"
    best = blend["best"]
    parts = ", ".join(
        f"{COMPONENT_NAMES.get(key, key)} {share:.0%}"
        for key, share in best["shares"].items()
        if share > 0
    )
    props = best["properties"]
    text = (
        f"Товарная смесь: {parts}"
        + (f", присадка {best['additive_kg_t']:.1f} кг/т" if best.get("additive") else "")
        + f"; сера {props['sulfur_mg_kg']:.1f} мг/кг, плотность {props['density_kg_m3']:.0f} кг/м³, "
        f"T95 {props['t95_c']:.0f} °C, цетановое число {props['cetane']:.1f} — в спецификации."
    )
    if blend.get("binding"):
        text += f" Ограничивает: {', '.join(blend['binding'])}."
    return text


def describe_budget(budget: dict | None, diesel_sulfur: float | None, base_limit: float) -> str:
    """What the tank's sulfur budget meant for this decision."""
    if not budget or budget.get("budget_mg_kg") is None:
        return ""
    limit = float(budget.get("limit_in_force_mg_kg") or budget["budget_mg_kg"])
    if diesel_sulfur is not None and diesel_sulfur > base_limit:
        return (
            f"Сера гидроочищенного ДТ {diesel_sulfur:.1f} мг/кг выше {base_limit:g}, но в пределах "
            f"серного бюджета резервуара {limit:.1f}: норму товарного продукта обеспечивает смесь."
        )
    return f"Серный бюджет резервуара: гидроочищенный ДТ до {limit:.1f} мг/кг."


def describe_economy(economy: dict | None) -> str:
    """The room to cool the reactor, as information for the technologist."""
    if not economy:
        return ""
    text = (
        f"Резерв для экономии: по модели отклика {economy['lever']} можно снизить с "
        f"{fmt(economy['from'])} до {fmt(economy['to'])} °C — сера ДТ составит "
        f"{economy['sulfur_mg_kg']:.1f} мг/кг при лимите {economy['limit_mg_kg']:.1f}"
    )
    if economy.get("needs_blend"):
        text += ", с разбавлением в смеси"
    return text + ". Это информация для технолога, а не рекомендация."


def describe_economics(effect: dict | None, *, is_hold: bool) -> str:
    """Output and specific energy, in operator language.

    On a hold there is nothing to compare, so the current level is stated instead of a
    change: the operator still needs to see what the regime costs.
    """
    if not effect:
        return ""
    out, energy = effect["output"], effect["energy"]
    if is_hold:
        return (
            f"Выпуск {fmt(out['from'])} {out['unit']}, индекс удельной энергии "
            f"{energy['from']:.2f} (1.0 — типичный режим обучающего периода)."
        )
    per_day = out["delta_t_day"]
    return (
        f"Выпуск {fmt(out['from'])} → {fmt(out['to'])} {out['unit']} "
        f"({out['pct']:+.1f} %, {per_day:+.0f} т/сут), "
        f"удельная энергия {energy['pct']:+.1f} % (индекс {energy['from']:.2f} → {energy['to']:.2f})."
    )


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
    economics: dict | None = None,
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
            describe_economics(economics, is_hold=True),
            _risk(best),
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
            describe_economics(economics, is_hold=False),
            _risk(best),
            f"Почему этот вариант: {why}." if why else "",
        ]
    lines += [SCOPE_NOTE, PRIORITY_NOTE]
    return " ".join(line for line in lines if line)
