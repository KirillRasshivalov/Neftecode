from neftecode.orchestration.orchestrator import Orchestrator


def test_orchestrator_normal_scenario_returns_recommendation():
    rec = Orchestrator().run_cycle(__import__("datetime").datetime(2024, 6, 1, 12, 0, 0), scenario="normal")
    assert rec.refuse is False
    # Stable scaffold state with no trigger: the regime is held, so there is no action.
    assert rec.audit["decision"]["outcome"] == "hold"
    assert rec.proposed_action is None
    assert rec.explanation


def test_orchestrator_stale_data_refuses():
    rec = Orchestrator().run_cycle(__import__("datetime").datetime(2025, 1, 10, 18, 0, 0), scenario="stale_data")
    assert rec.refuse is True
    assert rec.refuse_reason
