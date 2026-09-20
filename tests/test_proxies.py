"""Throughput and energy proxies, and the rule that quality may not be traded for output.

Round reference numbers, so every expectation is checkable by hand: the median regime is
T5 = 360 °C with the heat reference at 300 °C, so the temperature span is 60 °C, and the
gas term is 1.0 at the median. The index is then 0.7 + 0.3 = 1.0.
"""
from datetime import datetime

import pytest

from neftecode.agents.proxies import ProcessEconomics
from neftecode.domain.actions import ControlAction
from neftecode.domain.state import ProcessState, TagValue

T5, F26, F2, P13 = "242000:T5", "242000:F26", "242000:F2", "242000:P13"

REFERENCE = {
    "temperature_tag": T5,
    "feed_tag": F26,
    "gas_tag": F2,
    "pressure_tag": P13,
    "t5_median_c": 360.0,
    # (F2 / F26) * P13 at the regime below: (200 / 100) * 0.5 = 1.0
    "gas_term_median": 1.0,
    "heat_reference_c": 300.0,
    "w_heat": 0.7,
    "w_compression": 0.3,
    "density_t_m3": 0.85,
}

MEDIAN_REGIME = {T5: 360.0, F26: 100.0, F2: 200.0, P13: 0.5}


@pytest.fixture
def economics() -> ProcessEconomics:
    return ProcessEconomics.from_reference(REFERENCE)


def state(**overrides) -> ProcessState:
    values = {**MEDIAN_REGIME, **overrides}
    return ProcessState(
        timestamp=datetime(2026, 1, 1, 12, 0),
        controllable=dict(values),
        kip={tag: TagValue(tag=tag, value=value) for tag, value in values.items()},
    )


def move(**changes) -> ControlAction:
    return ControlAction(changes=changes, label="test")


# ------------------------------------------------------------------- the index


def test_the_index_is_one_at_the_median_regime(economics):
    assert economics.index(360.0, 200.0, 100.0, 0.5) == pytest.approx(1.0)


def test_holding_changes_nothing(economics):
    metrics = economics.metrics(state(), ControlAction(changes={}, label="noop"))
    assert metrics == {"throughput": pytest.approx(0.0), "energy_or_cost_proxy": pytest.approx(0.0)}


# -------------------------------------------------------------- one lever each


def test_a_hotter_reactor_costs_energy_and_leaves_output_alone(economics):
    # +6 °C on a 60 °C span is a tenth of the heat term, weighted 0.7 -> +7 % of the index.
    metrics = economics.metrics(state(), move(**{T5: 366.0}))
    assert metrics["energy_or_cost_proxy"] == pytest.approx(7.0)
    assert metrics["throughput"] == pytest.approx(0.0)


def test_more_recycle_gas_costs_energy_and_leaves_output_alone(economics):
    # +10 % on the gas term, weighted 0.3 -> +3 % of the index.
    metrics = economics.metrics(state(), move(**{F2: 220.0}))
    assert metrics["energy_or_cost_proxy"] == pytest.approx(3.0)
    assert metrics["throughput"] == pytest.approx(0.0)


def test_higher_pressure_costs_energy(economics):
    metrics = economics.metrics(state(), move(**{P13: 0.55}))
    assert metrics["energy_or_cost_proxy"] == pytest.approx(3.0)


def test_more_feed_raises_output_and_lowers_energy_per_unit(economics):
    # +25 % feed: the gas term falls to 0.8, so the index falls by 0.3 * 0.2 = 6 %.
    metrics = economics.metrics(state(), move(**{F26: 125.0}))
    assert metrics["throughput"] == pytest.approx(25.0)
    assert metrics["energy_or_cost_proxy"] == pytest.approx(-6.0)


# ------------------------------------------------------------------ the effect


def test_the_effect_states_output_in_tonnes_per_day_as_well(economics):
    effect = economics.effect(state(), move(**{F26: 105.0}))
    assert effect["output"]["pct"] == pytest.approx(5.0)
    # 5 m³/h * 0.85 t/m³ * 24 h
    assert effect["output"]["delta_t_day"] == pytest.approx(102.0)
    assert effect["energy"]["heater_delta"] == pytest.approx(0.0)
    assert effect["energy"]["compressor_delta"] < 0.0


def test_a_missing_lever_gives_no_numbers_rather_than_a_guess(economics):
    partial = ProcessState(timestamp=datetime(2026, 1, 1, 12, 0), controllable={T5: 360.0})
    assert economics.metrics(partial, move(**{T5: 362.0})) == {
        "throughput": 0.0,
        "energy_or_cost_proxy": 0.0,
    }
    assert economics.effect(partial, move(**{T5: 362.0})) is None


def test_a_cold_unit_gives_no_percentages(economics):
    # Below the heat reference the index approaches zero and a ratio stops meaning
    # anything; a stopped unit must not produce a confident-looking number.
    cold = state(**{T5: 295.0})
    assert economics.metrics(cold, move(**{T5: 297.0})) == {
        "throughput": 0.0,
        "energy_or_cost_proxy": 0.0,
    }
    assert economics.effect(cold, move(**{T5: 297.0})) is None
