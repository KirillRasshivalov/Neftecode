from datetime import datetime

import pytest

from neftecode.agents.reliability import ReliabilityAgentBaseline
from neftecode.domain.actions import ControlAction
from neftecode.domain.state import ProcessState

REFERENCE = {
    "envelope": {
        "242000:T5": {"p01": 347.0, "p50": 368.9, "p99": 387.9},
        "242000:F26": {"p01": 155.4, "p50": 255.4, "p99": 302.1},
        "242000:F2": {"p01": 80465.0, "p50": 93504.0, "p99": 106135.0},
        "242000:P13": {"p01": 3.61, "p50": 3.92, "p99": 4.03},
    },
    "t5_by_f26": [{"lo": 250.0, "hi": 260.0, "median": 370.7, "sigma": 6.7, "n": 15902}],
    "typical_f26": 255.4,
    "steps": {"242000:T5": 2.0, "242000:F26": 5.0, "242000:F2": 1000.0, "242000:P13": 0.05},
}
LEVERS = {"242000:T5": 370.7, "242000:F26": 255.4, "242000:F2": 93504.0, "242000:P13": 3.92}
AGENT = ReliabilityAgentBaseline(REFERENCE)


def make_state(running=True, **overrides):
    levers = dict(LEVERS)
    levers.update(overrides)
    return ProcessState(
        timestamp=datetime(2026, 7, 14, 16), controllable=levers, data_flags={"running": running}
    )


def test_a_stopped_unit_is_never_allowed():
    assessment = AGENT.assess(make_state(running=False))
    assert assessment.is_mode_allowed is False
    assert assessment.risk_class == "high"


def test_a_value_outside_the_training_envelope_is_not_allowed():
    assessment = AGENT.assess(make_state(**{"242000:T5": 390.0}))
    assert assessment.is_mode_allowed is False
    assert any("242000:T5" in factor for factor in assessment.risk_factors)


def test_typical_operation_is_low_risk_and_allowed():
    assessment = AGENT.assess(make_state())
    assert assessment.is_mode_allowed is True
    assert assessment.risk_class == "low"


def test_running_hotter_than_usual_for_the_feed_rate_raises_risk():
    typical = AGENT.assess(make_state()).risk_index
    hot = AGENT.assess(make_state(**{"242000:T5": 370.7 + 2.5 * 6.7}))
    assert hot.risk_index > typical
    assert any("σ" in factor for factor in hot.risk_factors)


def test_any_change_is_riskier_than_holding():
    state = make_state()
    hold = AGENT.assess(state, ControlAction(changes=dict(LEVERS))).risk_index
    changes = dict(LEVERS)
    changes["242000:T5"] += 2.0
    step = AGENT.assess(state, ControlAction(changes=changes)).risk_index
    assert step > hold


def test_missing_reference_band_is_reported_not_fatal():
    assessment = AGENT.assess(make_state(**{"242000:F26": 200.0}))
    assert any("Нет опорной статистики" in factor for factor in assessment.risk_factors)


def test_incomplete_reference_is_rejected():
    with pytest.raises(ValueError):
        ReliabilityAgentBaseline({"envelope": {}})


# ---------------------------------------------------------------------------
# The model range and the plant at its edge. Without a tolerance the decision chattered:
# a unit running just past p99 was "not allowed" one hour and allowed the next.

def test_the_current_regime_one_step_past_the_range_is_allowed_with_a_warning():
    assessment = AGENT.assess(make_state(**{"242000:P13": 4.06}))  # p99 4.03, step 0.05
    assert assessment.is_mode_allowed is True
    assert any("у границы" in factor for factor in assessment.risk_factors)


def test_the_current_regime_more_than_a_step_past_the_range_is_not_allowed():
    assessment = AGENT.assess(make_state(**{"242000:P13": 4.10}))
    assert assessment.is_mode_allowed is False
    assert any("больше чем на шаг" in factor for factor in assessment.risk_factors)


def test_a_step_may_not_move_a_lever_out_of_the_range():
    state = make_state(**{"242000:T5": 386.9})
    step_up = ControlAction(changes={"242000:T5": 388.9})  # p99 387.9
    assessment = AGENT.assess(state, step_up)
    assert assessment.is_mode_allowed is False
    assert any("выводит" in factor for factor in assessment.risk_factors)


def test_moving_another_lever_leaves_the_tolerated_one_alone():
    state = make_state(**{"242000:P13": 4.06})
    assert AGENT.assess(state, ControlAction(changes={"242000:T5": 372.7})).is_mode_allowed is True
