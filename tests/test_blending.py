"""Blending agent: mixing rules, the hard filter, the ranking and refusals.

Synthetic components with round numbers, so each expected value can be checked by hand
and the tests run on a clean clone without the lab-derived model file.
"""
import pytest

from neftecode.agents.blending import (
    Additive,
    BlendComponent,
    BlendingAgent,
    BlendSpec,
    build_spec,
)

SUMMER = BlendSpec(season="summer", sulfur_mg_kg_max=10.0, t95_c_max=360.0,
                   cetane_min=51.0, density_min=820.0, density_max=845.0)
WINTER = BlendSpec(season="winter", sulfur_mg_kg_max=10.0, t95_c_max=360.0,
                   cetane_min=49.0, density_min=800.0, density_max=845.0)
IMPROVER = Additive(key="A", label="цетаноповышающая", price_rel_per_t=100.0,
                    max_dose_kg_t=30.0, cetane_gain_max=10.0, dose_scale_kg_t=2.5)


def component(key, sulfur, density, t95, cetane, stock=2000.0):
    return BlendComponent(key=key, label=key, sulfur_mg_kg=sulfur, density_kg_m3=density,
                          t95_c=t95, cetane=cetane, stock_t=stock)


def standard(diesel_stock=2000.0, kerosene_stock=300.0, gas_oil_stock=400.0):
    return [
        component("hydrotreated_diesel", 8.6, 836.1, 347.0, 53.75, diesel_stock),
        component("kerosene", 5.0, 821.5, 290.0, 52.1, kerosene_stock),
        component("gas_oil", 8.0, 870.6, 349.0, 48.5, gas_oil_stock),
    ]


def agent(components=None, spec=SUMMER, additives=(IMPROVER,), batch=1000.0):
    return BlendingAgent(components or standard(), spec, list(additives), batch)


# ------------------------------------------------------------------ mixing


def test_sulfur_mixes_by_mass_and_density_by_volume():
    two = [component("hydrotreated_diesel", 10.0, 800.0, 300.0, 50.0),
           component("kerosene", 0.0, 1000.0, 300.0, 50.0)]
    candidate = agent(two)._evaluate({"hydrotreated_diesel": 0.5, "kerosene": 0.5}, two)
    assert candidate.properties["sulfur_mg_kg"] == pytest.approx(5.0)
    # 1 / (0.5/800 + 0.5/1000) = 888.9, not the mass-weighted 900
    assert candidate.properties["density_kg_m3"] == pytest.approx(888.889, abs=1e-3)


def test_distillation_and_cetane_mix_by_volume_share():
    two = [component("hydrotreated_diesel", 5.0, 800.0, 300.0, 60.0),
           component("kerosene", 5.0, 1000.0, 400.0, 40.0)]
    candidate = agent(two)._evaluate({"hydrotreated_diesel": 0.5, "kerosene": 0.5}, two)
    # equal masses, so the lighter component takes 1000/1800 of the volume
    lighter = 1000.0 / 1800.0
    assert candidate.properties["t95_c"] == pytest.approx(300.0 * lighter + 400.0 * (1 - lighter), abs=1e-3)
    assert candidate.properties["cetane"] == pytest.approx(60.0 * lighter + 40.0 * (1 - lighter), abs=1e-3)


def test_shares_always_sum_to_one_and_respect_stock():
    result = agent(standard(diesel_stock=600.0)).blend()
    for candidate in [result.best, *result.alternatives]:
        assert sum(candidate.shares.values()) == pytest.approx(1.0)
        assert candidate.tonnes["hydrotreated_diesel"] <= 600.0 + 1e-6


# --------------------------------------------------------- choosing a blend


def test_an_on_spec_product_is_shipped_as_it_is():
    result = agent().blend()
    assert result.outcome == "blend"
    assert result.best.shares["hydrotreated_diesel"] == 1.0
    assert result.best.additive is None


def test_off_spec_diesel_is_diluted_with_low_sulfur_kerosene():
    result = agent().blend(diesel_sulfur_mg_kg=11.5)
    assert result.outcome == "blend"
    assert result.best.properties["sulfur_mg_kg"] <= 10.0
    assert result.best.shares["kerosene"] > 0.0
    assert "сера" in result.binding


def test_dilution_takes_no_more_than_it_needs():
    # 11.5 → 10 with 5 mg/kg kerosene needs 1.5/6.5 = 23 %, so the 5 % grid gives 25 %.
    assert agent().blend(diesel_sulfur_mg_kg=11.5).best.shares["kerosene"] == pytest.approx(0.25)


def test_the_expensive_additive_is_used_only_when_cetane_demands_it():
    low_cetane = [
        component("hydrotreated_diesel", 8.0, 836.0, 347.0, 49.0),
        component("kerosene", 5.0, 821.5, 290.0, 48.0),
    ]
    result = agent(low_cetane).blend()
    assert result.outcome == "blend"
    assert result.best.additive == "A"
    assert result.best.properties["cetane"] >= 51.0 - 1e-9
    assert result.best.cost_rel_per_t > 1.0


def test_winter_relaxes_the_cetane_limit_and_saves_the_additive():
    low_cetane = [component("hydrotreated_diesel", 8.0, 836.0, 347.0, 50.0)]
    assert agent(low_cetane, spec=SUMMER).blend().best.additive == "A"
    assert agent(low_cetane, spec=WINTER).blend().best.additive is None


def test_ranking_is_deterministic():
    first = agent().blend(diesel_sulfur_mg_kg=12.0).model_dump()
    second = agent().blend(diesel_sulfur_mg_kg=12.0).model_dump()
    assert first == second


# ---------------------------------------------------------------- refusals


def test_a_limit_nothing_can_meet_is_named_with_its_distance():
    only_gas_oil = [component("gas_oil", 8.0, 870.6, 349.0, 48.5)]
    result = agent(only_gas_oil).blend()
    assert result.outcome == "refuse"
    assert "плотность" in result.refuse_reason
    assert result.closest_infeasible is not None


def test_a_cetane_gap_the_additive_cannot_close_refuses():
    hopeless = [component("hydrotreated_diesel", 8.0, 836.0, 347.0, 38.0)]
    result = agent(hopeless).blend()
    assert result.outcome == "refuse"
    assert "цетановое число" in result.refuse_reason


def test_stocks_below_the_batch_refuse_before_any_mixing():
    result = agent(standard(diesel_stock=100.0, kerosene_stock=100.0, gas_oil_stock=100.0)).blend()
    assert result.outcome == "refuse"
    assert "Запасов не хватает" in result.refuse_reason
    assert result.n_candidates == 0


def test_additive_dose_inverts_its_own_response():
    dose = IMPROVER.dose_for(4.0)
    assert IMPROVER.gain(dose) == pytest.approx(4.0)
    assert IMPROVER.dose_for(10.0) is None


def test_the_season_picks_the_organisers_limits():
    config = {"season": "summer", "spec": {
        "sulfur_mg_kg_max": 10.0, "t95_c_max": 360.0,
        "cetane_min": {"summer": 51.0, "winter": 49.0},
        "density_kg_m3": {"summer": [820.0, 845.0], "winter": [800.0, 845.0]},
    }}
    assert build_spec(config).cetane_min == 51.0
    winter = build_spec(config, "winter")
    assert (winter.cetane_min, winter.density_min) == (49.0, 800.0)


# ---------------------------------------------------------------------------
# The sulfur budget the blending agent hands to the hydrotreating side.


def test_the_budget_is_what_the_diluent_can_carry_under_the_dilution_cap():
    budget = agent().sulfur_budget()
    # 70 % diesel + 30 % kerosene at 5 mg/kg, 0.5 mg/kg under the limit: (9.5 − 1.5) / 0.7
    assert budget.budget_mg_kg == pytest.approx(8.0 / 0.7, abs=1e-3)
    assert budget.shares["hydrotreated_diesel"] >= 0.7 - 1e-9


def test_the_dilution_cap_keeps_the_budget_from_parking_the_unit_s_product():
    loose = BlendingAgent(standard(), SUMMER, [IMPROVER], 1000.0, max_dilution_share=0.7)
    assert loose.sulfur_budget().budget_mg_kg > agent().sulfur_budget().budget_mg_kg


def test_without_any_diluent_the_budget_is_the_limit_less_the_margin():
    only_diesel = [component("hydrotreated_diesel", 8.6, 836.1, 347.0, 53.75)]
    assert agent(only_diesel).sulfur_budget().budget_mg_kg == pytest.approx(9.5)


def test_no_budget_when_no_blend_passes_without_additive():
    heavy = [component("hydrotreated_diesel", 8.0, 870.0, 347.0, 53.0)]  # too dense on its own
    budget = agent(heavy).sulfur_budget()
    assert budget.budget_mg_kg is None
    assert budget.reason
