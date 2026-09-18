"""What-if scenarios the organisers ask for: change the inputs, watch the answer change.

Their example: sulfur in the crude rises, the straight-run diesel carries more of it,
and the system has to compensate — through the hydrotreating regime or through the
blend. This runs one decision on real data at `--at`, once as measured and once under
the scenario, and then sets the two ways out side by side:

- path A, the reactor: the lever step the orchestrator recommends, and the blend it
  leaves the tank with;
- path B, the tank: keep the regime and let the blend absorb the change.

    python -m scripts.blend_scenario --at 2026-01-27T16:00 --feed-sulfur 1.25
    python -m scripts.blend_scenario --at 2026-01-27T16:00 --stock hydrotreated_diesel=300
    python -m scripts.blend_scenario --at 2026-01-27T16:00 --season winter --feed-sulfur 1.3

Scenario inputs are marked as such in the trace (`data_flags`), so a card never passes
a hypothetical feed off as a measurement.
"""
from __future__ import annotations

import argparse

import pandas as pd

from neftecode.agents import QualityAgentBaseline, ReliabilityAgentBaseline
from neftecode.data.config import load_constraints
from neftecode.orchestration.orchestrator import Orchestrator
from scripts import realdata as rd
from neftecode.orchestration.explain import fmt
from scripts.run_real import (
    RealStateBuilder,
    _moved,
    blend_line,
    frozen_whitelist,
    full_card,
    load_blending,
    load_classifier,
    load_model_file,
    one_line,
    outcome,
    use_utf8_stdout,
)


def parse_stocks(items: list[str]) -> dict[str, float]:
    stocks = {}
    for item in items:
        key, _, value = item.partition("=")
        stocks[key.strip()] = float(value)
    return stocks


def build(overrides: dict, season: str | None, stocks: dict[str, float]):
    quality_params = load_model_file("quality_baseline.json")
    reference = load_model_file("reliability_reference.json")
    classifier = load_classifier()
    blending = load_blending(season, stocks)
    orchestrator = Orchestrator(
        quality_agent=QualityAgentBaseline(quality_params, classifier=classifier),
        reliability_agent=ReliabilityAgentBaseline(reference),
        state_builder=RealStateBuilder(
            quality_params,
            classifier_features=classifier.features if classifier else None,
            overrides=overrides,
        ),
        artifacts_dir=rd.REPO_ROOT / "artifacts" / "scenario",
        whitelist=frozen_whitelist(reference),
        constraints_cfg=load_constraints(),
        blending_agent=blending,
    )
    return orchestrator, blending


def sulfur_of(quality: dict | None) -> float | None:
    return ((quality or {}).get("metrics") or {}).get("sulfur_mg_kg")


def main(argv: list[str] | None = None) -> None:
    use_utf8_stdout()
    parser = argparse.ArgumentParser(prog="blend_scenario", description=__doc__.splitlines()[0])
    parser.add_argument("--at", required=True, help="ISO timestamp of the decision")
    parser.add_argument("--feed-sulfur", type=float, help="scenario sulfur in the hydrotreater feed, %% mass")
    parser.add_argument("--season", choices=("summer", "winter"), help="product specification")
    parser.add_argument("--stock", action="append", default=[], metavar="COMPONENT=TONNES",
                        help="override a tank stock, e.g. hydrotreated_diesel=300")
    args = parser.parse_args(argv)
    t = pd.Timestamp(args.at).to_pydatetime()
    stocks = parse_stocks(args.stock)

    measured, _ = build({}, args.season, stocks)
    base = measured.run_cycle(t, scenario="measured")

    overrides = {}
    if args.feed_sulfur is not None:
        overrides["feed_sulfur_pct_scenario"] = args.feed_sulfur
    scenario_orch, blending = build(overrides, args.season, stocks)
    rec = scenario_orch.run_cycle(t, scenario="what_if")

    flags = rec.audit.get("state", {}).get("data_flags", {})
    print(f"Сценарий на {t:%Y-%m-%d %H:%M}")
    if args.feed_sulfur is not None:
        print(f"  сера в сырье: измерено {flags.get('feed_sulfur_pct')} % → сценарий {args.feed_sulfur} %")
    if stocks:
        print(f"  запасы: {stocks}")
    if args.season:
        print(f"  спецификация: {args.season}")
    print()
    print("  как есть   " + one_line(base))
    print("  сценарий   " + one_line(rec))
    print()
    print(full_card(rec))
    print()

    if blending is None or (outcome(rec) == "REFUSE" and not rec.audit.get("blend")):
        return
    hold_sulfur = sulfur_of(rec.audit.get("baseline_quality"))
    print("Два пути:")
    if outcome(rec) == "RECOMMEND":
        chosen_sulfur = sulfur_of(rec.expected_effect.get("quality"))
        step = ", ".join(f"{tag} {fmt(a)} → {fmt(b)}" for tag, (a, b) in _moved(rec).items())
        print(f"  А. реактор   {step}: сера ДТ {chosen_sulfur:.2f} мг/кг")
        print(f"               смесь: {blend_line(rec.audit['blend'])}")
    else:
        print("  А. реактор   оркестратор не рекомендует менять режим")
    tank = blending.blend(hold_sulfur).model_dump(mode="json")
    print(f"  Б. резервуар режим не менять, сера ДТ {hold_sulfur:.2f} мг/кг")
    print(f"               смесь: {blend_line(tank)}")
    print(f"\nJSON-трассы: {scenario_orch.artifacts_dir}")


if __name__ == "__main__":
    main()
