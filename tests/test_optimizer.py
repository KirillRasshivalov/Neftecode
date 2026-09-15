from datetime import datetime

from neftecode.agents.optimizer import OptimizerAgent
from neftecode.domain.agent_results import QualityAssessment, ReliabilityAssessment
from neftecode.domain.state import ProcessState
from neftecode.safety.constraints import HardConstraints

T5, F26 = "242000:T5", "242000:F26"
ONE_LEVER = {
    "controllable_parameters": [
        {"tag": T5, "range": {"min": 347.0, "max": 388.0, "assumption": True}, "deltas": [-2.0, 0.0, 2.0]}
    ]
}
TWO_LEVERS = {
    "controllable_parameters": [
        *ONE_LEVER["controllable_parameters"],
        {"tag": F26, "range": {"min": 155.0, "max": 302.0, "assumption": True}, "deltas": [-5.0, 0.0, 5.0]},
    ]
}
CONFIG = {
    "hard": {"sulfur_mg_kg_max": 10.0},
    "ranking_weights": {"quality_risk": 100.0, "throughput": 10.0, "energy_or_cost_proxy": 5.0, "equipment_risk": 20.0},
}
STATE = ProcessState(timestamp=datetime(2026, 1, 27, 16), controllable={T5: 368.9, F26: 255.4})


class FlatQuality:
    def assess(self, state, action=None):
        return QualityAssessment(metrics={"sulfur_mg_kg": 8.0}, risk_of_spec_breach=0.0)


class FlatReliability:
    def assess(self, state, action=None):
        return ReliabilityAssessment(risk_index=0.2, is_mode_allowed=True)


def make_optimizer(whitelist=ONE_LEVER, max_levers=None):
    hard = HardConstraints(CONFIG, whitelist=whitelist)
    return OptimizerAgent(
        FlatQuality(), FlatReliability(), constraints=hard, whitelist=whitelist,
        ranking_cfg=CONFIG, max_levers_per_action=max_levers,
    )


def test_on_a_tie_holding_the_current_regime_wins():
    ranked = make_optimizer().propose(STATE)
    assert len({s.score for s in ranked}) == 1, "the scenario must really be a tie"
    assert ranked[0].action.is_noop()


def test_by_default_each_candidate_moves_one_lever():
    candidates = make_optimizer(TWO_LEVERS).evaluate(STATE)
    assert len(candidates) == 1 + 2 + 2
    assert all(len(c.action.changes) <= 1 for c in candidates)


def test_combining_levers_is_opt_in():
    candidates = make_optimizer(TWO_LEVERS, max_levers=2).evaluate(STATE)
    assert len(candidates) == 1 + 2 + 2 + 4
    assert any(len(c.action.changes) == 2 for c in candidates)


def test_rejected_candidates_are_kept_after_the_feasible_ones():
    narrow = {"controllable_parameters": [{**ONE_LEVER["controllable_parameters"][0], "range": {"min": 360.0, "max": 369.0}}]}
    candidates = make_optimizer(narrow).evaluate(STATE)
    flags = [c.feasible for c in candidates]
    assert flags == sorted(flags, reverse=True), "feasible first, rejected last"
    assert not all(flags)
    assert all(c.score is None for c in candidates if not c.feasible)
    assert all(c.feasible for c in make_optimizer(narrow).propose(STATE))


def test_throughput_lowers_the_score():
    optimizer = make_optimizer()
    assert optimizer._score({"throughput": 1.0}) < optimizer._score({"throughput": 0.0})


def test_risk_and_energy_raise_the_score():
    optimizer = make_optimizer()
    assert optimizer._score({"quality_risk": 0.1}) > optimizer._score({})
    assert optimizer._score({"energy_or_cost_proxy": 1.0}) > optimizer._score({})
