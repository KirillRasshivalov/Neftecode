from datetime import datetime

from neftecode.data.state_builder import ProcessStateBuilder
from neftecode.domain.actions import ControlAction
from neftecode.domain.agent_results import QualityAssessment, ReliabilityAssessment
from neftecode.safety.constraints import HardConstraints


def test_sulfur_hard_limit():
    state = ProcessStateBuilder().build(datetime(2024, 1, 1))
    action = ControlAction(changes={})
    quality = QualityAssessment(metrics={"sulfur_mg_kg": 12.0})
    reliability = ReliabilityAssessment(is_mode_allowed=True)
    ok, reasons = HardConstraints().check_action(state, action, quality, reliability)
    assert not ok
    assert any("sulfur" in r for r in reasons)


def test_tag_out_of_range():
    state = ProcessStateBuilder().build(datetime(2024, 1, 1))
    action = ControlAction(changes={"PLACEHOLDER_AVT_TEMP": 999.0})
    quality = QualityAssessment(metrics={"sulfur_mg_kg": 5.0})
    reliability = ReliabilityAssessment(is_mode_allowed=True)
    ok, reasons = HardConstraints().check_action(state, action, quality, reliability)
    assert not ok
    assert any("PLACEHOLDER_AVT_TEMP" in r for r in reasons)
