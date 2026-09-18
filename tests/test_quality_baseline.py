from datetime import datetime

import pytest

from neftecode.agents.quality import BreachClassifier, QualityAgentBaseline
from neftecode.domain.actions import ControlAction
from neftecode.domain.state import ProcessState, QualityReading

PARAMS = {
    "alpha": 0.1,
    "resid_q05": -3.5,
    "resid_q95": 3.1,
    "ref_gap_hours": 24.0,
    "sulfur_median_train": 8.5,
}
LEVERS = {"242000:T5": 368.9, "242000:F26": 255.4, "242000:F2": 93500.0, "242000:P13": 3.92}
AGENT = QualityAgentBaseline(PARAMS)


def make_state(lab=9.0, age_hours=6.0, ewma=8.8, pak_healthy=True, with_lab=True,
               windows=None, running=True):
    quality = {}
    if with_lab:
        quality["sulfur_mg_kg"] = QualityReading(
            metric="sulfur_mg_kg", value=lab, unit="mg/kg", source="lims", age_minutes=age_hours * 60
        )
    flags = {"running": running, "pak_sulfur": {"healthy": pak_healthy}}
    if windows is not None:
        flags["feature_windows"] = windows
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


# ---------------------------------------------------------------------------
# Breach classifier. Two features are enough to pin the behaviour; the fitted
# model has 42. The analyzer coefficient is the large one, as in the real fit.

CLASSIFIER_MODEL = {
    "features": ["pak:sulfur|2h", "lab_ewma"],
    "mean": [8.0, 8.5],
    "scale": [1.0, 1.0],
    "coef": [1.2, 0.3],
    "intercept": -2.0,
    "auc_heldout": 0.72,
    "trigger_threshold": 0.235,
}
WINDOWS = {"pak:sulfur|2h": 10.4, "lab_ewma": 7.7}
CLASSIFIED = QualityAgentBaseline(PARAMS, classifier=BreachClassifier(CLASSIFIER_MODEL))


def risk(assessment):
    return assessment.risk_of_spec_breach


def source(assessment):
    return assessment.details["risk_source"]


def test_without_a_classifier_the_risk_is_the_interval_estimate():
    assessment = AGENT.assess(make_state(windows=WINDOWS))
    assert source(assessment) == "interval"
    assert risk(assessment) == assessment.details["risk_interval"]


def test_the_classifier_sets_the_risk_when_nothing_disqualifies_it():
    assessment = CLASSIFIED.assess(make_state(windows=WINDOWS))
    assert source(assessment) == "classifier"
    # Holding the regime has no what-if increment, so this is the model's own number.
    assert risk(assessment) == pytest.approx(assessment.details["risk_classifier_hold"])
    assert assessment.details["classifier_auc_heldout"] == 0.72
    assert "model:breach_classifier" in assessment.features_used


def test_a_frozen_analyzer_disqualifies_the_classifier():
    assessment = CLASSIFIED.assess(make_state(windows=WINDOWS, pak_healthy=False))
    assert source(assessment) == "interval"
    assert "анализатор" in assessment.details["classifier_unavailable"]


def test_a_stopped_unit_disqualifies_the_classifier():
    assert source(CLASSIFIED.assess(make_state(windows=WINDOWS, running=False))) == "interval"


def test_a_missing_feature_disqualifies_the_classifier_instead_of_guessing():
    assessment = CLASSIFIED.assess(make_state(windows={"pak:sulfur|2h": 10.4, "lab_ewma": None}))
    assert source(assessment) == "interval"
    assert assessment.details["risk_classifier_hold"] is None


def test_a_state_without_feature_windows_disqualifies_the_classifier():
    assert source(CLASSIFIED.assess(make_state())) == "interval"


def test_a_higher_analyzer_reading_raises_the_classifier_risk():
    hot = CLASSIFIED.assess(make_state(windows={"pak:sulfur|2h": 11.0, "lab_ewma": 7.7}))
    cool = CLASSIFIED.assess(make_state(windows={"pak:sulfur|2h": 6.0, "lab_ewma": 7.7}))
    assert risk(hot) > risk(cool)


def test_the_action_moves_the_risk_by_the_physics_increment_either_way():
    # The classifier is deaf to the levers, so a step must move the risk by exactly what
    # the physics layer says. If this drifts, hold_margin stops being calibrated.
    step = move("242000:T5", 2.0)
    plain = risk(AGENT.assess(make_state(windows=WINDOWS))) - risk(AGENT.assess(make_state(windows=WINDOWS), step))
    model = risk(CLASSIFIED.assess(make_state(windows=WINDOWS))) - risk(
        CLASSIFIED.assess(make_state(windows=WINDOWS), step)
    )
    assert model == pytest.approx(plain, abs=1e-4)


def test_the_classifier_does_not_touch_the_interval_the_hard_check_uses():
    assert interval(CLASSIFIED.assess(make_state(windows=WINDOWS))) == interval(
        AGENT.assess(make_state(windows=WINDOWS))
    )


def test_a_classifier_with_mismatched_arrays_is_rejected():
    with pytest.raises(ValueError, match="disagree"):
        BreachClassifier({**CLASSIFIER_MODEL, "coef": [1.0]})


def test_a_classifier_missing_a_block_is_rejected():
    with pytest.raises(ValueError, match="intercept"):
        BreachClassifier({k: v for k, v in CLASSIFIER_MODEL.items() if k != "intercept"})


def test_an_extreme_score_stays_a_probability():
    wild = QualityAgentBaseline(PARAMS, classifier=BreachClassifier({**CLASSIFIER_MODEL, "coef": [500.0, 0.0]}))
    assert 0.0 <= risk(wild.assess(make_state(windows=WINDOWS))) <= 1.0
