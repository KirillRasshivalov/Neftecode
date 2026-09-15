from datetime import datetime

import pytest

from neftecode.agents.quality import QualityAgentBaseline
from neftecode.domain.actions import ControlAction
from neftecode.domain.state import ProcessState, QualityReading

PARAMS = {
    "alpha": 0.1,
    "resid_q05": -3.5,
    "resid_q95": 3.1,
    "ref_gap_hours": 24.0,
    "sulfur_median_train": 8.5,
}
LEVERS = {"242000:T5": 368.9, "242000:F26": 255.4, "242000:F2": 93500.0, "242000:F15": 3397.0}
AGENT = QualityAgentBaseline(PARAMS)


def make_state(lab=9.0, age_hours=6.0, ewma=8.8, pak_healthy=True, with_lab=True):
    quality = {}
    if with_lab:
        quality["sulfur_mg_kg"] = QualityReading(
            metric="sulfur_mg_kg", value=lab, unit="mg/kg", source="lims", age_minutes=age_hours * 60
        )
    flags = {"running": True, "pak_sulfur": {"healthy": pak_healthy}}
    if ewma is not None:
        flags["lab_sulfur_ewma"] = ewma
    return ProcessState(
        timestamp=datetime(2026, 7, 14, 16), quality=quality, controllable=dict(LEVERS), data_flags=flags
    )


def move(tag, delta):
    changes = dict(LEVERS)
    changes[tag] += delta
    return ControlAction(changes=changes)


def sulfur(assessment):
    return assessment.metrics["sulfur_mg_kg"]


def interval(assessment):
    return assessment.details["intervals"]["sulfur_mg_kg"]


def test_prediction_is_the_smoothed_history_not_the_last_result():
    assert sulfur(AGENT.assess(make_state(lab=12.0, ewma=8.8))) == pytest.approx(8.8)


def test_without_history_one_result_moves_the_median_by_alpha():
    assert sulfur(AGENT.assess(make_state(lab=12.0, ewma=None))) == pytest.approx(8.5 + 0.1 * (12.0 - 8.5))


def test_interval_brackets_the_mean():
    i = interval(AGENT.assess(make_state()))
    assert i["p05"] < i["mean"] < i["p95"]


# Physics sanity checks. Feed rate first: it is set by the production plan, not in
# response to quality, so its sign is the least confounded (CLAUDE.md §11).
def test_more_feed_raises_predicted_sulfur():
    assert sulfur(AGENT.assess(make_state(), move("242000:F26", 5.0))) > sulfur(AGENT.assess(make_state()))


def test_hotter_reactor_lowers_predicted_sulfur():
    assert sulfur(AGENT.assess(make_state(), move("242000:T5", 2.0))) < sulfur(AGENT.assess(make_state()))


def test_more_gas_per_feed_lowers_predicted_sulfur():
    assert sulfur(AGENT.assess(make_state(), move("242000:F2", 1000.0))) < sulfur(AGENT.assess(make_state()))


def test_an_action_that_changes_nothing_equals_no_action():
    unchanged = ControlAction(changes=dict(LEVERS))
    assert sulfur(AGENT.assess(make_state(), unchanged)) == sulfur(AGENT.assess(make_state()))


def test_an_old_lab_result_widens_the_interval():
    fresh = interval(AGENT.assess(make_state(age_hours=6.0)))
    old = interval(AGENT.assess(make_state(age_hours=96.0)))
    assert old["p95"] - old["p05"] > fresh["p95"] - fresh["p05"]


def test_breach_risk_grows_with_the_prediction():
    low = AGENT.assess(make_state(ewma=7.0)).risk_of_spec_breach
    high = AGENT.assess(make_state(ewma=9.5)).risk_of_spec_breach
    assert 0.0 <= low < high <= 1.0


def test_unhealthy_analyzer_lowers_confidence():
    assert AGENT.assess(make_state(pak_healthy=False)).confidence < AGENT.assess(make_state()).confidence


def test_no_lab_result_falls_back_to_the_median_with_low_confidence():
    assessment = AGENT.assess(make_state(with_lab=False, ewma=None))
    assert sulfur(assessment) == pytest.approx(8.5)
    assert assessment.confidence < 0.3


def test_incomplete_params_are_rejected():
    with pytest.raises(ValueError):
        QualityAgentBaseline({"alpha": 0.1})
