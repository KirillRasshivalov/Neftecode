from datetime import datetime

from neftecode.data.state_builder import ProcessStateBuilder
from neftecode.domain.actions import ControlAction
from neftecode.domain.agent_results import MetricInterval, QualityAssessment, ReliabilityAssessment
from neftecode.safety.constraints import HardConstraints


def test_sulfur_hard_limit_uses_mean_without_interval():
    state = ProcessStateBuilder().build(datetime(2024, 1, 1))
    action = ControlAction(changes={})
    quality = QualityAssessment(metrics={"sulfur_mg_kg": 12.0})
    reliability = ReliabilityAssessment(is_mode_allowed=True)
    result = HardConstraints().check_action(state, action, quality, reliability)
    assert not result.ok
    assert any("сера" in r for r in result.reasons)
    assert result.margins["sulfur_mg_kg"] < 0


def test_sulfur_hard_limit_uses_p95_when_interval_present():
    state = ProcessStateBuilder().build(datetime(2024, 1, 1))
    action = ControlAction(changes={})
    quality = QualityAssessment(
        metrics={"sulfur_mg_kg": 9.0},
        intervals={"sulfur_mg_kg": MetricInterval(mean=9.0, p05=7.0, p95=11.5)},
    )
    reliability = ReliabilityAssessment(is_mode_allowed=True)
    result = HardConstraints().check_action(state, action, quality, reliability)
    assert not result.ok
    assert any("p95" in r for r in result.reasons)
    assert result.margins["sulfur_mg_kg"] == round(10.0 - 11.5, 4)


def test_p95_below_limit_passes_even_if_mean_near_limit():
    state = ProcessStateBuilder().build(datetime(2024, 1, 1))
    action = ControlAction(changes={})
    quality = QualityAssessment(
        metrics={"sulfur_mg_kg": 9.8},
        intervals={"sulfur_mg_kg": MetricInterval(mean=9.8, p05=8.0, p95=9.9)},
    )
    reliability = ReliabilityAssessment(is_mode_allowed=True)
    result = HardConstraints().check_action(state, action, quality, reliability)
    assert result.ok
    assert result.margins["sulfur_mg_kg"] > 0


def test_tag_out_of_range():
    state = ProcessStateBuilder().build(datetime(2024, 1, 1))
    action = ControlAction(changes={"PLACEHOLDER_AVT_TEMP": 999.0})
    quality = QualityAssessment(metrics={"sulfur_mg_kg": 5.0})
    reliability = ReliabilityAssessment(is_mode_allowed=True)
    ok, reasons = HardConstraints().check_action(state, action, quality, reliability)
    assert not ok
    assert any("PLACEHOLDER_AVT_TEMP" in r for r in reasons)
