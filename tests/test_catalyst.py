"""Catalyst cycle: detecting a change, reading the drift as of a moment, and the agent.

Synthetic daily series with round numbers, so every expectation can be checked by hand
and the tests need no data files.
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # `scripts` lives at the root

from scripts import realdata as rd  # noqa: E402
from neftecode.agents.reliability import ReliabilityAgentBaseline  # noqa: E402
from neftecode.domain.state import ProcessState  # noqa: E402

DAY0 = pd.Timestamp("2025-01-01")
EOR = 13.56


def series(values, start=DAY0):
    return pd.Series(np.asarray(values, dtype=float), index=pd.date_range(start, periods=len(values), freq="D"))


def ramp(days, start_c, per_month):
    return series(start_c + per_month * np.arange(days) / 30.0)


# ----------------------------------------------------------------- detection


def test_a_restart_after_which_the_reactor_runs_much_cooler_is_a_catalyst_change():
    hot = series([12.0] * 60)
    cold = series([-10.0] * 40, start=DAY0 + pd.Timedelta(days=90))
    daily = pd.concat([hot, cold])
    restart = DAY0 + pd.Timedelta(days=90)
    assert rd.catalyst_changes(daily, pd.DatetimeIndex([restart])) == [restart]


def test_an_ordinary_restart_is_not_a_catalyst_change():
    daily = series([5.0] * 120)
    assert rd.catalyst_changes(daily, pd.DatetimeIndex([DAY0 + pd.Timedelta(days=60)])) == []


# ------------------------------------------------------------------- status


def test_a_steady_drift_is_extrapolated_to_the_end_of_run_level():
    daily = ramp(360, 0.0, 1.0)                        # +1 °C a month from 0
    t = DAY0 + pd.Timedelta(days=361)
    st = rd.catalyst_status(daily, [], t, EOR)
    assert st["state"] == "ageing"
    assert st["drift_c_per_month"] == pytest.approx(1.0, abs=0.01)
    assert st["excess_now_c"] == pytest.approx(12.0, abs=0.1)
    assert st["days_to_eor"] == pytest.approx((EOR - 12.0) * 30, abs=3)


def test_wear_is_read_on_the_pessimistic_side():
    # A late jump the trend line has not caught up with still counts.
    daily = pd.concat([ramp(346, 0.0, 1.0), series([15.0] * 14, start=DAY0 + pd.Timedelta(days=346))])
    st = rd.catalyst_status(daily, [], DAY0 + pd.Timedelta(days=361), EOR)
    assert st["state"] == "eor_reached"


def test_a_fresh_catalyst_is_not_extrapolated():
    change = DAY0 + pd.Timedelta(days=200)
    daily = pd.concat([series([12.0] * 190), ramp(60, -10.0, 20.0).set_axis(
        pd.date_range(change, periods=60, freq="D"))])
    st = rd.catalyst_status(daily, [change], change + pd.Timedelta(days=40), EOR)
    assert st["state"] == "too_short"


def test_right_after_a_long_stop_the_cycle_is_not_guessed():
    # Whether the stop was a catalyst change is only known three weeks later, and the
    # status must not read that from the future.
    stop_end = DAY0 + pd.Timedelta(days=200)
    daily = ramp(260, 0.0, 1.0)
    st = rd.catalyst_status(daily, [stop_end], stop_end + pd.Timedelta(days=5), EOR, long_stops=[stop_end])
    assert st["state"] == "start_up"


def test_nothing_after_the_decision_moment_is_used():
    daily = ramp(400, 0.0, 1.0)
    t = DAY0 + pd.Timedelta(days=200)
    early = rd.catalyst_status(daily, [], t, EOR)
    truncated = rd.catalyst_status(daily.loc[: t - pd.Timedelta(days=1)], [], t, EOR)
    assert early == truncated


# -------------------------------------------------------------------- agent

REFERENCE = {
    "envelope": {"242000:T5": {"p01": 347.0, "p50": 368.9, "p99": 387.9}},
    "t5_by_f26": [{"lo": 250.0, "hi": 260.0, "median": 370.7, "sigma": 6.7, "n": 15902}],
    "typical_f26": 255.4,
    "steps": {"242000:T5": 2.0},
}
AGENT = ReliabilityAgentBaseline(REFERENCE)


def state_with(catalyst):
    return ProcessState(
        timestamp=datetime(2026, 3, 1, 16),
        controllable={"242000:T5": 370.7, "242000:F26": 255.4},
        data_flags={"running": True, "catalyst": catalyst},
    )


def test_near_the_end_of_run_the_agent_warns_and_asks_not_to_heat():
    cat = {"state": "ageing", "days_to_eor": 30, "drift_c_per_month": 1.0, "excess_now_c": 12.5, "eor_excess_c": EOR}
    assessment = AGENT.assess(state_with(cat))
    assert any("Катализатор" in f for f in assessment.risk_factors)
    assert any("не повышать" in c for c in assessment.soft_constraints)
    assert assessment.details["catalyst"] == cat


def test_far_from_the_end_of_run_the_card_stays_quiet():
    cat = {"state": "ageing", "days_to_eor": 200, "drift_c_per_month": 1.0, "excess_now_c": 7.0, "eor_excess_c": EOR}
    assert not any("Катализатор" in f for f in AGENT.assess(state_with(cat)).risk_factors)


def test_catalyst_life_does_not_move_the_severity_index():
    base = AGENT.assess(state_with(None)).risk_index
    spent = {"state": "eor_reached", "days_to_eor": 0, "drift_c_per_month": 1.0, "excess_now_c": 15.0, "eor_excess_c": EOR}
    assert AGENT.assess(state_with(spent)).risk_index == base
