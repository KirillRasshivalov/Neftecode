import copy

import pytest
import re
from datetime import datetime

from neftecode.agents.blending import BlendComponent, BlendingAgent, BlendSpec
from neftecode.data.config import load_constraints
from neftecode.domain.agent_results import QualityAssessment, ReliabilityAssessment
from neftecode.domain.state import ProcessState, QualityReading
from neftecode.orchestration.orchestrator import Orchestrator

T = datetime(2026, 1, 27, 16)
T5 = "242000:T5"
WHITELIST = {
    "controllable_parameters": [
        {"tag": T5, "range": {"min": 347.0, "max": 388.0, "assumption": True}, "deltas": [-2.0, 0.0, 2.0]}
    ]
}


class LeverQuality:
    """Predicted sulfur falls by 0.5 mg/kg per °C of extra T5; risk rises above 8 mg/kg."""

    def __init__(self, base: float, sensitivity: float = 0.5) -> None:
        self.base = base
        self.sensitivity = sensitivity

    def assess(self, state, action=None):
        delta = 0.0
        if action is not None and T5 in action.changes:
            delta = action.changes[T5] - state.controllable[T5]
        sulfur = self.base - self.sensitivity * delta
        risk = max(0.0, min(1.0, (sulfur - 8.0) / 4.0))
        return QualityAssessment(metrics={"sulfur_mg_kg": sulfur}, risk_of_spec_breach=risk, confidence=0.7)


class SourcedQuality(LeverQuality):
    """LeverQuality that also reports which estimate produced the probability.

    The two estimates are on different scales, so the orchestrator has to pick the
    matching threshold rather than compare them against one number.
    """

    def __init__(self, base: float, risk_source: str, risk: float) -> None:
        super().__init__(base)
        self.risk_source = risk_source
        self.risk = risk

    def assess(self, state, action=None):
        delta = 0.0
        if action is not None and T5 in action.changes:
            delta = action.changes[T5] - state.controllable[T5]
        return QualityAssessment(
            metrics={"sulfur_mg_kg": self.base - self.sensitivity * delta},
            risk_of_spec_breach=max(0.0, min(1.0, self.risk - 0.05 * delta)),
            confidence=0.7,
            details={"risk_source": self.risk_source},
        )


class FlatReliability:
    def assess(self, state, action=None):
        return ReliabilityAssessment(risk_index=0.2, risk_class="low", is_mode_allowed=True)


class LeverReliability:
    """Severity rises with T5, so a hotter reactor always costs reliability."""

    def __init__(self, base: float = 0.45) -> None:
        self.base = base

    def assess(self, state, action=None):
        delta = 0.0
        if action is not None and T5 in action.changes:
            delta = action.changes[T5] - state.controllable[T5]
        index = max(0.0, self.base + 0.075 * delta)
        risk_class = "low" if index < 0.4 else ("medium" if index < 0.7 else "high")
        return ReliabilityAssessment(risk_index=round(index, 4), risk_class=risk_class, is_mode_allowed=True)


class FixedBuilder:
    def __init__(self, state):
        self.state = state

    def build(self, timestamp, scenario=None):
        return self.state


def make_state(lab=8.0, running=True, age_minutes=360.0):
    return ProcessState(
        timestamp=T,
        controllable={T5: 368.9},
        quality={
            "sulfur_mg_kg": QualityReading(
                metric="sulfur_mg_kg", value=lab, unit="mg/kg", source="lims", age_minutes=age_minutes
            )
        },
        data_flags={
            "complete": True,
            "running": running,
            "running_detail": None if running else "242000:F26 = 3.2 (порог 50)",
        },
    )


def make_orchestrator(tmp_path, state, quality=None, hold_margin=None, reliability=None, blending=None):
    cfg = copy.deepcopy(load_constraints())
    if hold_margin is not None:
        cfg.setdefault("decision", {})["hold_margin"] = hold_margin
    return Orchestrator(
        quality_agent=quality or LeverQuality(base=8.0),
        reliability_agent=reliability or FlatReliability(),
        state_builder=FixedBuilder(state),
        artifacts_dir=tmp_path,
        whitelist=WHITELIST,
        constraints_cfg=cfg,
        blending_agent=blending,
    )


def outcome_of(rec):
    """Assert the outcome invariant, then return the outcome."""
    outcome = rec.audit["decision"]["outcome"]
    if outcome == "recommend":
        assert rec.refuse is False and rec.proposed_action is not None and rec.refuse_reason is None
    elif outcome == "hold":
        assert rec.refuse is False and rec.proposed_action is None and rec.refuse_reason is None
    else:
        assert outcome == "refuse"
        assert rec.refuse is True and rec.proposed_action is None and rec.refuse_reason
    assert rec.explanation
    return outcome


def test_default_constructor_keeps_working(tmp_path):
    rec = Orchestrator(artifacts_dir=tmp_path).run_cycle(datetime(2024, 6, 1, 12), scenario="normal")
    assert outcome_of(rec) == "hold"


def test_no_problem_means_hold_even_if_a_change_looks_better(tmp_path):
    rec = make_orchestrator(tmp_path, make_state(lab=9.0), LeverQuality(base=9.0)).run_cycle(T)
    assert outcome_of(rec) == "hold"
    assert "признаков проблемы нет" in rec.audit["decision"]["why"]
    assert rec.audit["decision"]["best_candidate"] != "noop", "a better-looking change existed"


def test_off_spec_lab_with_a_worthwhile_step_recommends_it(tmp_path):
    rec = make_orchestrator(tmp_path, make_state(lab=11.0), LeverQuality(base=9.5)).run_cycle(T)
    assert outcome_of(rec) == "recommend"
    assert rec.proposed_action.changes == {T5: 370.9}
    assert any("вне спецификации" in t for t in rec.audit["decision"]["triggers"])


def test_a_problem_without_a_worthwhile_step_holds(tmp_path):
    orchestrator = make_orchestrator(tmp_path, make_state(lab=11.0), LeverQuality(base=9.5), hold_margin=1000.0)
    rec = orchestrator.run_cycle(T)
    assert outcome_of(rec) == "hold"
    assert "проблема есть" in rec.audit["decision"]["why"]


def test_an_off_spec_current_regime_is_fixed_regardless_of_the_margin(tmp_path):
    orchestrator = make_orchestrator(tmp_path, make_state(lab=11.0), LeverQuality(base=10.5), hold_margin=1000.0)
    rec = orchestrator.run_cycle(T)
    assert outcome_of(rec) == "recommend"
    assert rec.proposed_action.changes == {T5: 370.9}


def test_nothing_feasible_refuses_and_asks_for_a_human(tmp_path):
    rec = make_orchestrator(tmp_path, make_state(lab=12.0), LeverQuality(base=12.0)).run_cycle(T)
    assert outcome_of(rec) == "refuse"
    assert "человека" in rec.refuse_reason


def test_candidates_come_from_the_injected_whitelist(tmp_path):
    rec = make_orchestrator(tmp_path, make_state(lab=11.0), LeverQuality(base=9.5)).run_cycle(T)
    assert set(rec.proposed_action.changes) | {t for alt in rec.alternatives for t in alt.action.changes} <= {T5}


def test_refuses_when_the_unit_is_not_running(tmp_path):
    rec = make_orchestrator(tmp_path, make_state(running=False)).run_cycle(T)
    assert outcome_of(rec) == "refuse"
    assert "установка не работает" in rec.refuse_reason
    assert "242000:F26" in rec.refuse_reason


def test_a_shutdown_is_reported_before_stale_data(tmp_path):
    rec = make_orchestrator(tmp_path, make_state(running=False, age_minutes=10_000)).run_cycle(T)
    assert "установка не работает" in rec.refuse_reason


def test_explanation_has_no_raw_floats(tmp_path):
    rec = make_orchestrator(tmp_path, make_state(lab=11.0), LeverQuality(base=9.5)).run_cycle(T)
    assert not re.search(r"\d+\.\d{5,}", rec.explanation)
    # The card says where the 10 mg/kg limit really applies, not just where we check it.
    assert "гидроочищенного ДТ" in rec.explanation
    assert "товарной смеси" in rec.explanation


def test_every_decision_writes_a_trace(tmp_path):
    make_orchestrator(tmp_path, make_state()).run_cycle(T)
    assert list(tmp_path.glob("recommendation_*.json"))


def test_a_classifier_probability_below_the_interval_threshold_still_triggers(tmp_path):
    # 0.30 is under the interval threshold of 0.50 but over the classifier's 0.235.
    quality = SourcedQuality(base=9.0, risk_source="classifier", risk=0.30)
    rec = make_orchestrator(tmp_path, make_state(lab=9.0), quality).run_cycle(T)
    assert outcome_of(rec) == "recommend"
    assert rec.audit["decision"]["breach_risk_source"] == "classifier"
    assert rec.audit["decision"]["breach_risk_threshold"] == 0.2512
    assert any("классификатор" in trigger for trigger in rec.audit["decision"]["triggers"])


def test_the_same_probability_from_the_interval_estimate_does_not_trigger(tmp_path):
    quality = SourcedQuality(base=9.0, risk_source="interval", risk=0.30)
    rec = make_orchestrator(tmp_path, make_state(lab=9.0), quality).run_cycle(T)
    assert outcome_of(rec) == "hold"
    assert rec.audit["decision"]["breach_risk_threshold"] == 0.5
    assert rec.audit["decision"]["triggers"] == []


def test_elevated_severity_alone_is_not_answered_by_raising_severity(tmp_path):
    # A hotter reactor lowers sulfur and so scores well, but the only complaint here is
    # that the equipment is working hard. Working it harder is not an answer.
    rec = make_orchestrator(
        tmp_path, make_state(lab=9.0), LeverQuality(base=9.0), reliability=LeverReliability()
    ).run_cycle(T)
    assert outcome_of(rec) == "hold"
    assert rec.audit["trigger_kinds"] == ["reliability"]
    assert rec.audit["equipment_only_filter"]["applied"] is True
    assert rec.audit["equipment_only_filter"]["dropped"] > 0


def test_an_off_spec_lab_lifts_the_severity_filter(tmp_path):
    # With quality in trouble the trade is legitimate again: sulfur comes first.
    rec = make_orchestrator(
        tmp_path, make_state(lab=11.0), LeverQuality(base=9.5), reliability=LeverReliability()
    ).run_cycle(T)
    assert outcome_of(rec) == "recommend"
    assert "lab_off_spec" in rec.audit["trigger_kinds"]
    assert "equipment_only_filter" not in rec.audit
    assert rec.proposed_action.changes[T5] > make_state().controllable[T5]


def make_blending():
    spec = BlendSpec(season="summer", sulfur_mg_kg_max=10.0, t95_c_max=360.0,
                     cetane_min=51.0, density_min=820.0, density_max=845.0)
    components = [
        BlendComponent(key="hydrotreated_diesel", label="ДТ", sulfur_mg_kg=8.6,
                       density_kg_m3=836.1, t95_c=347.0, cetane=53.75, stock_t=2000.0),
        BlendComponent(key="kerosene", label="керосин", sulfur_mg_kg=5.0,
                       density_kg_m3=821.5, t95_c=290.0, cetane=52.1, stock_t=300.0),
    ]
    return BlendingAgent(components, spec, [], 1000.0)


def test_with_a_blending_agent_the_card_carries_the_commercial_blend(tmp_path):
    rec = make_orchestrator(
        tmp_path, make_state(lab=9.0), LeverQuality(base=9.0), blending=make_blending()
    ).run_cycle(T)
    assert rec.expected_effect["blend"]["outcome"] == "blend"
    assert any("товарная смесь" in label for label in rec.constraints_checked)
    assert "Товарная смесь" in rec.explanation


def test_the_tank_budget_turns_a_refusal_into_a_step_the_tank_can_finish(tmp_path):
    # At 11.5 mg/kg no single step reaches 10, and without blending that is a refusal.
    # The tank absorbs up to 11.43 (30 % kerosene at 5 mg/kg, 0.5 mg/kg margin), so a
    # step to 10.5 is enough: the reactor moves once and the blend does the rest.
    rec = make_orchestrator(
        tmp_path, make_state(lab=9.0), LeverQuality(base=11.5), blending=make_blending()
    ).run_cycle(T)
    assert outcome_of(rec) == "recommend"
    assert rec.audit["sulfur_budget"]["limit_in_force_mg_kg"] == pytest.approx(11.429, abs=1e-3)
    assert any("11.429" in label for label in rec.constraints_checked)
    blend = rec.expected_effect["blend"]
    assert blend["best"]["properties"]["sulfur_mg_kg"] <= 10.0


def test_without_blending_the_same_regime_is_still_refused(tmp_path):
    rec = make_orchestrator(tmp_path, make_state(lab=9.0), LeverQuality(base=11.5)).run_cycle(T)
    assert outcome_of(rec) == "refuse"


def test_diesel_above_ten_inside_the_budget_is_not_forced_to_move(tmp_path):
    # Without the budget 10.4 fails the hard check and the orchestrator must move.
    # With it the regime is admissible, and the card says the blend covers the gap.
    rec = make_orchestrator(
        tmp_path, make_state(lab=9.0), LeverQuality(base=10.4), blending=make_blending(), hold_margin=1000.0
    ).run_cycle(T)
    assert outcome_of(rec) == "hold"
    assert "серного бюджета" in rec.explanation


def test_an_on_spec_hold_reports_room_to_cool_the_reactor_as_information(tmp_path):
    rec = make_orchestrator(
        tmp_path, make_state(lab=8.0), LeverQuality(base=8.0), blending=make_blending()
    ).run_cycle(T)
    assert outcome_of(rec) == "hold"
    assert rec.proposed_action is None                     # information, not an action
    economy = rec.audit["economy"]
    # 0.5 mg/kg per °C: 8.0 → 9.0 → 10.0 → 11.0 fits 11.43, 12.0 does not — three steps.
    assert economy["to"] == pytest.approx(368.9 - 6.0)
    assert economy["needs_blend"] is True
    assert "не рекомендация" in rec.explanation


def test_without_a_blending_agent_the_card_is_unchanged(tmp_path):
    rec = make_orchestrator(tmp_path, make_state(lab=9.0), LeverQuality(base=9.0)).run_cycle(T)
    assert rec.expected_effect.get("blend") is None
    assert not any("товарная смесь" in label for label in rec.constraints_checked)
