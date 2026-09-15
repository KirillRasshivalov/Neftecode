from datetime import datetime

from neftecode.data.config import load_constraints
from neftecode.domain.agent_results import QualityAssessment, ReliabilityAssessment
from neftecode.domain.state import ProcessState, QualityReading
from neftecode.orchestration.orchestrator import Orchestrator

T = datetime(2026, 1, 27, 16)
WHITELIST = {
    "controllable_parameters": [
        {"tag": "242000:T5", "range": {"min": 347.0, "max": 388.0, "assumption": True}, "deltas": [-2.0, 0.0, 2.0]}
    ]
}


class FlatQuality:
    def assess(self, state, action=None):
        return QualityAssessment(metrics={"sulfur_mg_kg": 8.0}, risk_of_spec_breach=0.0, confidence=0.7)


class FlatReliability:
    def assess(self, state, action=None):
        return ReliabilityAssessment(risk_index=0.2, risk_class="low", is_mode_allowed=True)


class FixedBuilder:
    def __init__(self, state):
        self.state = state

    def build(self, timestamp, scenario=None):
        return self.state


def make_state(running=True, age_minutes=360.0):
    return ProcessState(
        timestamp=T,
        controllable={"242000:T5": 368.9},
        quality={
            "sulfur_mg_kg": QualityReading(
                metric="sulfur_mg_kg", value=8.0, unit="mg/kg", source="lims", age_minutes=age_minutes
            )
        },
        data_flags={
            "complete": True,
            "running": running,
            "running_detail": None if running else "242000:F26 = 3.2 (порог 50)",
        },
    )


def make_orchestrator(tmp_path, state):
    return Orchestrator(
        quality_agent=FlatQuality(),
        reliability_agent=FlatReliability(),
        state_builder=FixedBuilder(state),
        artifacts_dir=tmp_path,
        whitelist=WHITELIST,
        constraints_cfg=load_constraints(),
    )


def test_default_constructor_keeps_working(tmp_path):
    rec = Orchestrator(artifacts_dir=tmp_path).run_cycle(datetime(2024, 6, 1, 12), scenario="normal")
    assert rec.refuse is False


def test_a_stable_state_holds_instead_of_changing_something(tmp_path):
    rec = make_orchestrator(tmp_path, make_state()).run_cycle(T)
    assert rec.refuse is False
    assert rec.proposed_action is not None and rec.proposed_action.is_noop()


def test_the_injected_whitelist_drives_the_candidates(tmp_path):
    rec = make_orchestrator(tmp_path, make_state()).run_cycle(T)
    tags = set(rec.proposed_action.changes) | {t for alt in rec.alternatives for t in alt.action.changes}
    assert tags <= {"242000:T5"}


def test_refuses_when_the_unit_is_not_running(tmp_path):
    rec = make_orchestrator(tmp_path, make_state(running=False)).run_cycle(T)
    assert rec.refuse is True
    assert "установка не работает" in rec.refuse_reason
    assert "242000:F26" in rec.refuse_reason


def test_a_shutdown_is_reported_before_stale_data(tmp_path):
    rec = make_orchestrator(tmp_path, make_state(running=False, age_minutes=10_000)).run_cycle(T)
    assert "установка не работает" in rec.refuse_reason


def test_every_decision_writes_a_trace(tmp_path):
    make_orchestrator(tmp_path, make_state()).run_cycle(T)
    assert list(tmp_path.glob("recommendation_*.json"))
