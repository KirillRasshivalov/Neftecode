from datetime import datetime

from neftecode.agents.optimizer import OptimizerAgent
from neftecode.domain.agent_results import QualityAssessment, ReliabilityAssessment
from neftecode.domain.state import ProcessState
from neftecode.safety.constraints import HardConstraints

WHITELIST = {
    "controllable_parameters": [
        {"tag": "242000:T5", "range": {"min": 347.0, "max": 388.0, "assumption": True}, "deltas": [-2.0, 0.0, 2.0]}
    ]
}
CONFIG = {
    "hard": {"sulfur_mg_kg_max": 10.0},
    "ranking_weights": {"quality_risk": 100.0, "throughput": 10.0, "energy_or_cost_proxy": 5.0, "equipment_risk": 20.0},
}
STATE = ProcessState(timestamp=datetime(2026, 1, 27, 16), controllable={"242000:T5": 368.9})


class FlatQuality:
    def assess(self, state, action=None):
        return QualityAssessment(metrics={"sulfur_mg_kg": 8.0}, risk_of_spec_breach=0.0)


class FlatReliability:
    def assess(self, state, action=None):
        return ReliabilityAssessment(risk_index=0.2, is_mode_allowed=True)


def make_optimizer():
    hard = HardConstraints(CONFIG, whitelist=WHITELIST)
    return OptimizerAgent(FlatQuality(), FlatReliability(), constraints=hard, whitelist=WHITELIST, ranking_cfg=CONFIG)


def test_on_a_tie_holding_the_current_regime_wins():
    ranked = make_optimizer().propose(STATE)
    assert len({s.score for s in ranked}) == 1, "the scenario must really be a tie"
    assert ranked[0].action.is_noop()


def test_throughput_lowers_the_score():
    optimizer = make_optimizer()
    assert optimizer._score({"throughput": 1.0}) < optimizer._score({"throughput": 0.0})


def test_risk_and_energy_raise_the_score():
    optimizer = make_optimizer()
    assert optimizer._score({"quality_risk": 0.1}) > optimizer._score({})
    assert optimizer._score({"energy_or_cost_proxy": 1.0}) > optimizer._score({})


def test_candidates_come_from_the_injected_whitelist():
    ranked = make_optimizer().propose(STATE)
    assert len(ranked) == 3
    assert {tag for s in ranked for tag in s.action.changes} <= {"242000:T5"}
