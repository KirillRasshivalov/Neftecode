import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neftecode.demo.scenarios import DEMO_SCENARIOS
from neftecode.domain.recommendation import OperatorRecommendation
from neftecode.orchestration.orchestrator import Orchestrator

SHOWCASE_DIR = Path(__file__).resolve().parent / "showcase"
ASSETS = Path(__file__).resolve().parent / "assets"

WEEK_META = {
    "stable": {
        "title": "Устойчивый режим",
        "blurb": "Система не вмешивается без нужды — оператор видит HOLD.",
    },
    "risk": {
        "title": "Риск по качеству",
        "blurb": "Есть признак проблемы — формируется безопасная рекомендация.",
    },
    "degraded": {
        "title": "Деградация данных",
        "blurb": "Установка работает, а источники качества отказывают.",
    },
    "shutdown": {
        "title": "Останов установки",
        "blurb": "Режимные советы бессмысленны — честный отказ.",
    },
}


def inject_css() -> None:
    css = (ASSETS / "theme.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def outcome_of(rec: dict[str, Any] | OperatorRecommendation) -> str:
    if isinstance(rec, OperatorRecommendation):
        data = rec.model_dump(mode="json")
    else:
        data = rec
    return (data.get("outcome") or data.get("audit", {}).get("decision", {}).get("outcome") or (
        "refuse" if data.get("refuse") else ("recommend" if data.get("proposed_action") else "hold")
    )).lower()


def badge(outcome: str) -> str:
    labels = {
        "recommend": ("RECOMMEND", "badge-recommend"),
        "hold": ("HOLD", "badge-hold"),
        "refuse": ("REFUSE", "badge-refuse"),
    }
    text, cls = labels.get(outcome, (outcome.upper(), "badge-hold"))
    return f'<span class="nc-badge {cls}">{text}</span>'


def fmt_num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(v) >= 1000:
        return f"{v:,.0f}".replace(",", " ")
    return f"{v:.{digits}f}"


def load_showcase(name: str) -> dict[str, Any]:
    path = SHOWCASE_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run_live(scenario: str) -> OperatorRecommendation:
    sc = DEMO_SCENARIOS[scenario]
    return Orchestrator().run_cycle(sc.timestamp, scenario=scenario)


def try_real_available() -> bool:
    cache = ROOT / "data" / "cache"
    models = ROOT / "models"
    needed = ["quality_baseline.json", "reliability_reference.json"]
    return cache.exists() and any(cache.glob("*.parquet")) and all((models / n).exists() for n in needed)


def hero() -> None:
    st.markdown(
        """
        <div class="nc-hero">
          <div class="nc-hero-glow"></div>
          <div class="nc-kicker">Мультиагентная система · АВТ → гидроочистка → блендинг</div>
          <h1 class="nc-brand">НЕФТЕКОД</h1>
          <p class="nc-tagline">
            Безопасные, объяснимые рекомендации оператору.<br/>
            Качество и жёсткие ограничения важнее экономики.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pipeline(rec: dict[str, Any]) -> None:
    outcome = outcome_of(rec)
    steps = [
        ("Состояние", "Телеметрия + ЛИМС/ПАК"),
        ("Качество", "Прогноз и риск спеки"),
        ("Надёжность", "Тяжесть режима"),
        ("Оптимизатор", "Допустимые шаги"),
        ("Оркестратор", outcome.upper()),
    ]
    cells = []
    for i, (title, sub) in enumerate(steps):
        active = "is-active" if i == len(steps) - 1 else ""
        cells.append(
            f'<div class="nc-pipe-step {active}"><div class="nc-pipe-title">{title}</div>'
            f'<div class="nc-pipe-sub">{sub}</div></div>'
        )
        if i < len(steps) - 1:
            cells.append('<div class="nc-pipe-arrow">›</div>')
    st.markdown(f'<div class="nc-pipeline">{"".join(cells)}</div>', unsafe_allow_html=True)


def metric_chips(rec: dict[str, Any]) -> None:
    state = rec.get("audit", {}).get("state", {})
    quality = state.get("quality", {}).get("sulfur_mg_kg", {}) or {}
    effect_q = (rec.get("expected_effect") or {}).get("quality") or {}
    sulfur = effect_q.get("metrics", {}).get("sulfur_mg_kg")
    if sulfur is None:
        sulfur = quality.get("value")
    interval = (effect_q.get("details") or {}).get("intervals", {}).get("sulfur_mg_kg") or {}
    if not interval and effect_q.get("intervals"):
        interval = effect_q["intervals"].get("sulfur_mg_kg") or {}
    p95 = interval.get("p95") if isinstance(interval, dict) else None
    risk = effect_q.get("risk_of_spec_breach")
    conf = rec.get("confidence")
    reliability = (rec.get("expected_effect") or {}).get("reliability") or {}
    lab_age = quality.get("age_minutes")
    lab_age_h = None if lab_age is None else float(lab_age) / 60.0
    pak = state.get("data_flags", {}).get("pak_sulfur") or {}
    pak_label = "исправен" if pak.get("healthy") is True else (
        "нет данных" if not pak else "неисправен / заморожен"
    )

    cols = st.columns(5)
    chips = [
        ("Сера · прогноз", f"{fmt_num(sulfur, 2)} мг/кг", "Лаборатория и модель раздельно"),
        ("Сера · p95", f"{fmt_num(p95, 2)} мг/кг", "Жёсткая проверка идёт по p95"),
        ("Риск спеки", f"{0 if risk is None else risk:.0%}", (effect_q.get("details") or {}).get("risk_source", "—")),
        ("Уверенность", f"{0 if conf is None else conf:.0%}", f"ЛИМС: {fmt_num(lab_age_h, 1)} ч"),
        ("ПАК", pak_label, "Признак, не замена анализа"),
    ]
    for col, (label, value, hint) in zip(cols, chips):
        with col:
            st.markdown(
                f'<div class="nc-chip"><div class="nc-chip-label">{label}</div>'
                f'<div class="nc-chip-value">{value}</div><div class="nc-chip-hint">{hint}</div></div>',
                unsafe_allow_html=True,
            )
    if reliability:
        st.caption(
            f"Тяжесть режима: **{reliability.get('risk_class', '—')}** "
            f"({fmt_num(reliability.get('risk_index'), 2)})"
        )


def render_action(rec: dict[str, Any]) -> None:
    outcome = outcome_of(rec)
    action = rec.get("proposed_action")
    st.markdown('<div class="nc-section-title">Предлагаемое действие</div>', unsafe_allow_html=True)
    if outcome == "refuse":
        st.markdown(
            f'<div class="nc-panel nc-panel-refuse"><strong>Рекомендации нет.</strong><br/>'
            f'{rec.get("refuse_reason") or "Система отказалась от рискованного совета."}</div>',
            unsafe_allow_html=True,
        )
        return
    if outcome == "hold" or not action or not action.get("changes"):
        st.markdown(
            '<div class="nc-panel nc-panel-hold"><strong>Режим не менять.</strong><br/>'
            "Нет признака проблемы, достаточно сильного чтобы оправдать вмешательство, "
            "либо удержание лучше допустимых шагов.</div>",
            unsafe_allow_html=True,
        )
        return
    state = rec.get("audit", {}).get("state", {})
    controllable = state.get("controllable") or {}
    rows = []
    for tag, new_v in action.get("changes", {}).items():
        old_v = controllable.get(tag)
        unit = ((state.get("kip") or {}).get(tag) or {}).get("unit") or ""
        rows.append(
            f"<tr><td>{tag}</td><td>{fmt_num(old_v)} → <strong>{fmt_num(new_v)}</strong></td>"
            f"<td>{unit}</td></tr>"
        )
    st.markdown(
        f'<div class="nc-panel nc-panel-recommend"><table class="nc-table"><thead>'
        f"<tr><th>Параметр</th><th>Сейчас → рекомендация</th><th>Ед.</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def render_agents(rec: dict[str, Any]) -> None:
    st.markdown('<div class="nc-section-title">Мультиагентный разбор</div>', unsafe_allow_html=True)
    q = (rec.get("expected_effect") or {}).get("quality") or {}
    r = (rec.get("expected_effect") or {}).get("reliability") or {}
    blend = (rec.get("expected_effect") or {}).get("blend")
    audit = rec.get("audit") or {}
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="nc-agent"><div class="nc-agent-name">Агент качества</div>', unsafe_allow_html=True)
        st.write(f"Прогноз серы: **{fmt_num((q.get('metrics') or {}).get('sulfur_mg_kg'))} мг/кг**")
        st.write(f"Риск нарушения: **{(q.get('risk_of_spec_breach') or 0):.0%}**")
        source = (q.get("details") or {}).get("risk_source")
        st.caption(f"Источник риска: {source or '—'}")
        terms = (q.get("details") or {}).get("classifier_terms") or []
        if terms:
            st.caption("Вклады классификатора: " + ", ".join(
                f"{t.get('feature', t)}" if isinstance(t, dict) else str(t) for t in terms[:3]
            ))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="nc-agent"><div class="nc-agent-name">Агент надёжности</div>', unsafe_allow_html=True)
        st.write(f"Класс: **{r.get('risk_class', '—')}** · индекс {fmt_num(r.get('risk_index'))}")
        st.write(f"Режим допустим: **{'да' if r.get('is_mode_allowed', True) else 'нет'}**")
        factors = r.get("risk_factors") or []
        if factors:
            st.caption(factors[0])
        cat = (audit.get("state") or {}).get("data_flags", {}).get("catalyst") or {}
        cat_state = cat.get("state")
        if cat_state in ("start_up", "too_short"):
            st.caption("Катализатор: не оценивается")
        elif cat_state == "ageing" and cat.get("days_to_eor") is not None:
            months = float(cat["days_to_eor"]) / 30.0
            st.caption(f"Катализатор: ≈ {months:.0f} мес до уровня замены")
        elif cat_state:
            st.caption(f"Катализатор: {cat_state}")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="nc-agent"><div class="nc-agent-name">Агент смеси</div>', unsafe_allow_html=True)
        if not blend:
            st.write("Блок смеси не подключён в этом прогоне.")
            st.caption("На реальных данных появляется товарный рецепт и серный бюджет.")
        elif blend.get("outcome") != "blend":
            st.write(blend.get("refuse_reason") or "Допустимой смеси нет")
        else:
            best = blend.get("best") or {}
            props = best.get("properties") or {}
            st.write(
                f"Сера смеси **{fmt_num(props.get('sulfur_mg_kg'))}** · "
                f"цетан **{fmt_num(props.get('cetane'), 1)}** · "
                f"T95 **{fmt_num(props.get('t95_c'), 0)} °C**"
            )
            shares = best.get("shares") or {}
            st.caption(
                " / ".join(f"{k}: {v:.0%}" for k, v in shares.items() if v)
            )
        budget = audit.get("sulfur_budget")
        if budget:
            st.caption(
                f"Лимит режима {fmt_num(budget.get('limit_in_force_mg_kg'))} мг/кг — "
                "бюджет от агента смеси, не «ослабленный ГОСТ»."
            )
        st.markdown("</div>", unsafe_allow_html=True)


def render_constraints_and_explain(rec: dict[str, Any]) -> None:
    left, right = st.columns((1, 1.2))
    with left:
        st.markdown('<div class="nc-section-title">Проверка ограничений</div>', unsafe_allow_html=True)
        items = rec.get("constraints_checked") or []
        html = "".join(f"<li>{item}</li>" for item in items)
        st.markdown(f'<ul class="nc-checks">{html}</ul>', unsafe_allow_html=True)
        economy = (rec.get("audit") or {}).get("economy")
        if economy:
            st.markdown(
                f'<div class="nc-note"><strong>Информация об экономии</strong> — не рекомендация.<br/>'
                f'Можно охладить {economy.get("lever")}: '
                f'{fmt_num(economy.get("from"))} → {fmt_num(economy.get("to"))}, '
                f'оставаясь в бюджете серы.</div>',
                unsafe_allow_html=True,
            )
    with right:
        st.markdown('<div class="nc-section-title">Объяснение оператору</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="nc-explain">{rec.get("explanation") or "—"}</div>',
            unsafe_allow_html=True,
        )
        triggers = (rec.get("audit") or {}).get("trigger_kinds") or []
        decision = (rec.get("audit") or {}).get("decision") or {}
        if decision.get("why"):
            st.caption(f"Почему так: {decision['why']}")
        if triggers:
            st.caption("Признаки: " + ", ".join(triggers))
        elif outcome_of(rec) == "hold":
            st.caption("Признаков проблемы нет — система сознательно не трогает режим.")


def render_card(rec: dict[str, Any], *, title: str) -> None:
    outcome = outcome_of(rec)
    ts = rec.get("timestamp", "—")
    st.markdown(
        f'<div class="nc-card-head"><div><div class="nc-card-kicker">Карточка оператора</div>'
        f'<h2 class="nc-card-title">{title}</h2>'
        f'<div class="nc-card-meta">{ts}</div></div>{badge(outcome)}</div>',
        unsafe_allow_html=True,
    )
    if rec.get("problem_or_risk"):
        st.markdown(
            f'<div class="nc-problem"><span>Проблема / риск</span>'
            f'<p>{rec["problem_or_risk"]}</p></div>',
            unsafe_allow_html=True,
        )
    pipeline(rec)
    metric_chips(rec)
    render_action(rec)
    render_agents(rec)
    render_constraints_and_explain(rec)
    with st.expander("Полный JSON трассы (для проверки жюри)"):
        st.json(rec)


def main() -> None:
    st.set_page_config(
        page_title="НЕФТЕКОД · консоль оператора",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    hero()

    with st.sidebar:
        st.markdown("### Режим показа")
        mode = st.radio(
            "Источник",
            [
                "Продающее демо (готовые карточки)",
                "Живой цикл (каркас)",
            ],
            index=0,
        )
        st.markdown("---")
        if mode.startswith("Продающее"):
            week = st.selectbox(
                "Демо-неделя",
                list(WEEK_META),
                format_func=lambda k: f"{WEEK_META[k]['title']} · {k}",
            )
            st.caption(WEEK_META[week]["blurb"])
            run = st.button("Показать карточку", type="primary", use_container_width=True)
        else:
            scenario = st.selectbox(
                "Сценарий каркаса",
                list(DEMO_SCENARIOS),
                format_func=lambda k: f"{k} — {DEMO_SCENARIOS[k].description[:48]}…",
            )
            run = st.button("Запустить цикл", type="primary", use_container_width=True)
            week = None

        st.markdown("---")
        real_ok = try_real_available()
        st.markdown("#### Реальные данные")
        if real_ok:
            st.success("Кэш и модели найдены — можно подключить `scripts.run_real`.")
        else:
            st.info(
                "Сейчас нет `data/cache/*.parquet` и/или JSON в `models/`. "
                "UI работает на showcase и живом каркасе — этого достаточно для продажи архитектуры."
            )
        st.markdown(
            '<div class="nc-side-foot">Приоритет: качество → безопасность → объяснимость → экономика</div>',
            unsafe_allow_html=True,
        )

    if "rec" not in st.session_state:
        st.session_state.rec = load_showcase("stable")
        st.session_state.title = WEEK_META["stable"]["title"]

    if run:
        if mode.startswith("Продающее"):
            st.session_state.rec = load_showcase(week)
            st.session_state.title = WEEK_META[week]["title"]
        else:
            live = run_live(scenario)
            st.session_state.rec = live.model_dump(mode="json")
            st.session_state.title = f"Живой цикл · {scenario}"

    render_card(st.session_state.rec, title=st.session_state.title)

    st.markdown(
        """
        <div class="nc-footer">
          НЕФТЕКОД · прототип МАС для товарного дизеля ·
          рекомендация или честный отказ · воспроизводимый audit trail
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
