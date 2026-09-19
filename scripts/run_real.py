"""Run the decision cycle on real data: a bridge until the real ProcessStateBuilder lands.

Builds a `ProcessState` from `data/cache` (sentinels stripped, unit-prefixed tags,
LIMS usable 4 h after sampling, ПАК with a health flag) and runs the orchestrator
with the baseline agents. Kirill's `src/neftecode/data/` is not touched; this file
shows what his builder needs to produce.

Once, to build the model files:

    python -m scripts.analysis.labels
    python -m scripts.analysis.quality_baseline_fit
    python -m scripts.analysis.reliability_reference

Then:

    python -m scripts.run_real --week stable       # one decision per day at 16:00
    python -m scripts.run_real --at 2026-07-14T16:00
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

import pandas as pd

import yaml

from neftecode.agents import BreachClassifier, QualityAgentBaseline, ReliabilityAgentBaseline
from neftecode.agents.blending import BlendingAgent, build_additives, build_components, build_spec
from neftecode.data.config import load_constraints
from neftecode.domain.recommendation import OperatorRecommendation
from neftecode.domain.state import ProcessState, QualityReading, TagValue
from neftecode.orchestration.explain import fmt
from neftecode.orchestration.orchestrator import Orchestrator
from scripts import realdata as rd
from scripts.analysis.quality_baseline_fit import ewma_after, usable_labels

#: Recommendation cadence. The organisers ask for a step of 15 to 60 minutes; telemetry
#: is on a 10-minute grid, so any multiple of 10 works. At 60 minutes a demo week is 168
#: decisions, which is why a week prints its transitions rather than every line.
DEFAULT_EVERY_MINUTES = 60


def load_classifier() -> BreachClassifier | None:
    """The breach classifier, if it was fitted **and** passed its acceptance checks.

    Optional on purpose. Without the file — or with a run that failed the checks in
    `scripts/analysis/breach_classifier.py` — the quality agent keeps the interval
    estimate it used before, and nothing else in the pipeline changes.
    """
    path = rd.MODELS_DIR / "breach_classifier.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("accepted"):
        failed = [k for k, ok in (report.get("acceptance_checks") or {}).items() if not ok]
        print(f"классификатор не принят ({', '.join(failed) or 'нет отчёта'}) — остаётся оценка по интервалу")
        return None
    model = report.get("linear_model")
    return BreachClassifier(model) if model else None


def load_blending(season: str | None = None, stocks: dict[str, float] | None = None) -> BlendingAgent | None:
    """The blending agent, if the component properties were derived from the lab.

    `season` and `stocks` override `configs/blending.yaml` — the scenario inputs the
    organisers want to be able to change.
    """
    path = rd.MODELS_DIR / "blend_components.json"
    if not path.exists():
        print("нет models/blend_components.json — блок смеси пропущен "
              "(python -m scripts.analysis.blend_components)")
        return None
    derived = json.loads(path.read_text(encoding="utf-8"))
    config = yaml.safe_load((rd.REPO_ROOT / "configs" / "blending.yaml").read_text(encoding="utf-8"))
    for key, tonnes in (stocks or {}).items():
        config["components"][key]["stock_t"] = float(tonnes)
    return BlendingAgent(
        build_components(derived, config),
        build_spec(config, season),
        build_additives(config),
        config["batch_t"],
        config.get("grid_step_pct", 5),
    )


def load_model_file(name: str) -> dict:
    path = rd.MODELS_DIR / name
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. Run once: python -m scripts.analysis.labels && "
            "python -m scripts.analysis.quality_baseline_fit && "
            "python -m scripts.analysis.reliability_reference"
        )
    return json.loads(path.read_text(encoding="utf-8"))


class RealStateBuilder:
    """`ProcessState` at any timestamp, using only what was available at that time."""

    def __init__(
        self,
        quality_params: dict,
        classifier_features: list[str] | None = None,
        overrides: dict | None = None,
    ) -> None:
        # Scenario inputs laid over the measured state, e.g. a feed sulfur the crude does
        # not really have. They are marked as such in `data_flags` so no card can pass a
        # scenario off as a measurement.
        self.overrides = dict(overrides or {})
        self.tele = rd.load_telemetry()
        self.labels = pd.read_parquet(rd.MODELS_DIR / "labels.parquet").sort_values("sampled_at", kind="stable")
        self.labels = self.labels.reset_index(drop=True)
        self.pak = rd.load_pak_sulfur()
        self.feed = rd.load_feed_sulfur().dropna(subset=["feed_sulfur_pct"])
        self.feed["available_at"] = self.feed["sampled_at"] + rd.LIMS_DELAY
        self.feed_median = float(self.feed["feed_sulfur_pct"].median())

        usable = usable_labels(self.labels)
        self.usable = usable
        self.usable_available = usable["available_at"].reset_index(drop=True)
        self.usable_ewma = ewma_after(
            usable["sulfur_mg_kg"].to_numpy(),
            float(quality_params["alpha"]),
            float(quality_params["sulfur_median_train"]),
        )
        self.running_labels = (
            self.labels[self.labels["running_at_sample"] == True]  # noqa: E712
            .sort_values("sampled_at", kind="stable").reset_index(drop=True)
        )

        # Trailing window means, only when something asks for them: this is the stand-in
        # for the history window `ProcessState` does not carry yet (extension E4).
        self.windows = None
        if classifier_features:
            wanted = set(classifier_features)
            frame = rd.window_means(rd.load_feature_frame())
            self.windows = frame[[c for c in frame.columns if c in wanted]]

    def _feed_sulfur(self, t: pd.Timestamp) -> float:
        """Latest feed sulfur the lab had published by `t`, % mass; the median if none yet.

        Feed samples are sparse (132 in 3.5 years), so this is often weeks old. It only
        anchors a scenario ratio, never a prediction on its own.
        """
        pos = int(self.feed["available_at"].searchsorted(t, side="right")) - 1
        return round(float(self.feed["feed_sulfur_pct"].iloc[pos]), 4) if pos >= 0 else self.feed_median

    def _feature_windows(self, t: pd.Timestamp) -> dict[str, float | None] | None:
        """Model inputs as of `t`: trailing telemetry means plus lab history.

        A missing value stays `None` rather than being filled, so the classifier
        disqualifies itself instead of scoring a guess.
        """
        if self.windows is None:
            return None
        times = pd.DatetimeIndex([t])
        row = rd.asof_matrix(self.windows, times)[0]
        history = rd.lab_history(self.running_labels, self.usable, self.usable_ewma, times).iloc[0]
        out = {name: _number(value) for name, value in zip(self.windows.columns, row)}
        out.update({name: _number(value) for name, value in history.items()})
        return out

    def build(self, timestamp: datetime, scenario: str | None = None) -> ProcessState:
        t = pd.Timestamp(timestamp)
        pos = rd.asof_position(self.tele.index, t)
        if pos is None:
            raise ValueError(f"no telemetry at or before {t}")
        row = self.tele.iloc[pos]

        controllable: dict[str, float] = {}
        kip: dict[str, TagValue] = {}
        missing: list[str] = []
        for tag, value in row.items():
            if pd.isna(value):
                missing.append(tag)
                continue
            spec = rd.LEVERS[tag.split(":", 1)[1]]
            controllable[tag] = float(value)
            kip[tag] = TagValue(tag=tag, value=float(value), unit=str(spec["unit"]), source="242000_tags")

        f26, t5 = row[rd.tag_id("F26")], row[rd.tag_id("T5")]
        running = None if pd.isna(f26) or pd.isna(t5) else bool(
            f26 > rd.RUNNING_MIN["F26"] and t5 > rd.RUNNING_MIN["T5"]
        )
        running_detail = None
        if running is False:
            running_detail = (
                f"{rd.tag_id('F26')} = {f26:.4g} (порог {rd.RUNNING_MIN['F26']:g}), "
                f"{rd.tag_id('T5')} = {t5:.4g} (порог {rd.RUNNING_MIN['T5']:g})"
            )

        quality: dict[str, QualityReading] = {}
        lab = rd.lab_asof(self.labels, t)
        if lab is not None:
            quality["sulfur_mg_kg"] = QualityReading(
                metric="sulfur_mg_kg",
                value=float(lab["sulfur_mg_kg"]),
                unit="mg/kg",
                source="lims",
                measured_at=lab["sampled_at"].to_pydatetime(),
                age_minutes=round((t - lab["sampled_at"]).total_seconds() / 60.0, 1),
            )
        k = int(self.usable_available.searchsorted(t, side="right")) - 1
        ewma = round(float(self.usable_ewma[k]), 4) if k >= 0 else None

        return ProcessState(
            timestamp=t.to_pydatetime(),
            kip=kip,
            quality=quality,
            controllable=controllable,
            data_flags={
                "scaffold_mode": False,
                "source": "scripts.run_real bridge over data/cache",
                "telemetry_at": self.tele.index[pos].isoformat(),
                "complete": not missing,
                "missing_tags": missing,
                "running": running,
                "running_detail": running_detail,
                "lab_sulfur_ewma": ewma,
                "pak_sulfur": rd.pak_status(self.pak, t),
                "feature_windows": self._feature_windows(t),
                "feed_sulfur_pct": self._feed_sulfur(t),
                **self.overrides,
                "scenario": scenario,
            },
            notes=[
                "Мост scripts/run_real.py: заглушки удалены, теги с префиксом установки, "
                "ЛИМС доступен через 4 ч после отбора.",
            ],
        )


def _number(value: object) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), 6)


def frozen_whitelist(reference: dict) -> dict:
    """The four frozen levers: model range = training envelope p01–p99, one step either way."""
    items = []
    for short, spec in rd.LEVERS.items():
        tag = rd.tag_id(short)
        env = reference["envelope"][tag]
        step = float(spec["step"])
        items.append({
            "tag": tag,
            "unit": spec["unit"],
            "description": spec["label"],
            "current_source": "242000_tags",
            "range": {"min": env["p01"], "max": env["p99"], "assumption": True},
            "deltas": [-step, 0.0, step],
        })
    return {"controllable_parameters": items, "state_tags": {"avt": [], "hydrotreating": [], "blending": []}}


def outcome(rec: OperatorRecommendation) -> str:
    if rec.refuse:
        return "REFUSE"
    if rec.proposed_action is None or rec.proposed_action.is_noop():
        return "HOLD"
    return "RECOMMEND"


def _interval(quality: dict | None) -> dict:
    return ((quality or {}).get("details") or {}).get("intervals", {}).get("sulfur_mg_kg", {})


RISK_LABELS = {"classifier": "классификатор", "interval": "оценка по интервалу"}
COMPONENTS = {"hydrotreated_diesel": "ДТ", "kerosene": "керосин", "gas_oil": "газойль"}


def blend_line(blend: dict) -> str:
    if blend.get("outcome") != "blend" or not blend.get("best"):
        return "допустимой нет — " + (blend.get("refuse_reason") or "")
    best = blend["best"]
    shares = " + ".join(
        f"{COMPONENTS.get(k, k)} {w:.0%}" for k, w in best["shares"].items() if w > 0
    )
    props = best["properties"]
    additive = f" + присадка {best['additive_kg_t']:.1f} кг/т" if best.get("additive") else ""
    binding = f" · ограничивает: {', '.join(blend['binding'])}" if blend.get("binding") else ""
    return (
        f"{shares}{additive} · сера {props['sulfur_mg_kg']:.1f} · плотность {props['density_kg_m3']:.0f} · "
        f"T95 {props['t95_c']:.0f} · ЦЧ {props['cetane']:.1f} · цена {best['cost_rel_per_t']:.3f}{binding}"
    )


def _risk_label(quality: dict | None) -> str:
    details = (quality or {}).get("details") or {}
    return RISK_LABELS.get(details.get("risk_source"), "оценка")


def _moved(rec: OperatorRecommendation) -> dict[str, tuple[float, float]]:
    current = rec.audit.get("state", {}).get("controllable", {})
    if rec.proposed_action is None:
        return {}
    return {
        tag: (current[tag], new)
        for tag, new in rec.proposed_action.changes.items()
        if tag in current and abs(new - current[tag]) > 1e-9
    }


def one_line(rec: OperatorRecommendation) -> str:
    state = rec.audit.get("state", {})
    flags = state.get("data_flags", {})
    lab = state.get("quality", {}).get("sulfur_mg_kg")
    base = _interval(rec.audit.get("baseline_quality"))
    lab_text = f"ЛИМС {lab['value']:.1f} ({lab['age_minutes'] / 60:.0f} ч)" if lab else "ЛИМС —"
    p95 = f"p95 {base['p95']:.1f}" if base else "p95 —"
    running = {True: "работает", False: "стоит", None: "?"}[flags.get("running")]
    action = ", ".join(f"{tag} {fmt(old)}→{fmt(new)}" for tag, (old, new) in _moved(rec).items())
    return (
        f"{rec.timestamp:%Y-%m-%d %H:%M}  {outcome(rec):<9}  {lab_text:<18} {p95:<9} "
        f"установка {running}  {action}"
    ).rstrip()


def full_card(rec: OperatorRecommendation) -> str:
    state = rec.audit.get("state", {})
    flags = state.get("data_flags", {})
    current = state.get("controllable", {})
    lab = state.get("quality", {}).get("sulfur_mg_kg")
    pak = flags.get("pak_sulfur") or {}
    decision = rec.audit.get("decision", {})
    base = _interval(rec.audit.get("baseline_quality"))

    lines = [f"═══ {rec.timestamp:%Y-%m-%d %H:%M} · {outcome(rec)} ═══"]
    lines.append("1 Состояние   " + " · ".join(f"{tag} {fmt(v)}" for tag, v in current.items()))
    lab_text = f"ЛИМС {lab['value']:.1f} мг/кг ({lab['age_minutes'] / 60:.0f} ч назад)" if lab else "ЛИМС нет"
    lines.append(f"              {lab_text} · сглаженный уровень {fmt(flags.get('lab_sulfur_ewma'))} мг/кг")
    if pak:
        state_text = "исправен" if pak.get("healthy") else "неисправен/заморожен"
        lines.append(f"              ПАК {pak['value']:.1f} ppm ({state_text}) — признак, не результат анализа")
    lines.append(f"2 Проблема    {rec.problem_or_risk}")

    if outcome(rec) == "RECOMMEND":
        lines.append("3 Действие    " + ", ".join(f"{tag} {fmt(old)} → {fmt(new)}" for tag, (old, new) in _moved(rec).items()))
    elif outcome(rec) == "HOLD":
        lines.append("3 Действие    режим не менять")
    else:
        lines.append("3 Действие    — (отказ)")

    if outcome(rec) != "REFUSE":
        quality = rec.expected_effect.get("quality", {})
        reliability = rec.expected_effect.get("reliability", {})
        after = _interval(quality)
        lines.append(
            f"4 Эффект      сера {fmt(base.get('mean'))} → {fmt(after.get('mean'))} мг/кг, "
            f"p95 {fmt(base.get('p95'))} → {fmt(after.get('p95'))} · "
            f"риск нарушения {quality.get('risk_of_spec_breach', 0.0):.0%} "
            f"({_risk_label(quality)}) · "
            f"тяжесть {reliability.get('risk_class', '?')} ({reliability.get('risk_index', 0.0):.2f})"
        )
    lines.append("5 Проверки    " + "; ".join(rec.constraints_checked))
    lines.append("6 Уверенность " + (f"{rec.confidence:.2f}" if rec.confidence is not None else "—"))
    lines.append(f"7 Почему      {rec.explanation}")
    blend = rec.expected_effect.get("blend") or rec.audit.get("blend")
    if blend:
        lines.append("8 Смесь       " + blend_line(blend))
    if decision:
        lines.append(f"  Решение     {decision.get('outcome')}: {decision.get('why')}")
    lines.append(f"  Вариантов   {rec.audit.get('n_candidates', '—')}, допустимых {rec.audit.get('n_feasible', '—')}")
    return "\n".join(lines)


def decision_times(week: str, every_minutes: int = DEFAULT_EVERY_MINUTES) -> list[pd.Timestamp]:
    start = rd.DEMO_WEEKS[week]
    step = pd.Timedelta(minutes=every_minutes)
    return [start + step * i for i in range(int(pd.Timedelta(days=7) / step))]


def transitions(recs: list[OperatorRecommendation]) -> list[OperatorRecommendation]:
    """Only the decisions that differ from the one before, so a week stays readable."""
    out, previous = [], None
    for rec in recs:
        key = (outcome(rec), tuple(sorted(_moved(rec))))
        if key != previous:
            out.append(rec)
            previous = key
    return out


def use_utf8_stdout() -> None:
    """The card draws box rules and arrows; a Windows console defaults to cp1251."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass


def main(argv: list[str] | None = None) -> list[OperatorRecommendation]:
    use_utf8_stdout()
    parser = argparse.ArgumentParser(prog="run_real", description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--week", choices=sorted(rd.DEMO_WEEKS), help="frozen demo week")
    group.add_argument("--at", help="ISO timestamp")
    parser.add_argument(
        "--every", type=int, default=DEFAULT_EVERY_MINUTES,
        help=f"minutes between decisions in a week (default {DEFAULT_EVERY_MINUTES})",
    )
    parser.add_argument("--all-cards", action="store_true", help="print the full card for every decision")
    args = parser.parse_args(argv)

    quality_params = load_model_file("quality_baseline.json")
    reference = load_model_file("reliability_reference.json")
    classifier = load_classifier()
    blending = load_blending()
    orchestrator = Orchestrator(
        quality_agent=QualityAgentBaseline(quality_params, classifier=classifier),
        reliability_agent=ReliabilityAgentBaseline(reference),
        state_builder=RealStateBuilder(
            quality_params, classifier_features=classifier.features if classifier else None
        ),
        artifacts_dir=rd.REPO_ROOT / "artifacts" / "real",
        whitelist=frozen_whitelist(reference),
        constraints_cfg=load_constraints(),
        blending_agent=blending,
    )

    times = decision_times(args.week, args.every) if args.week else [pd.Timestamp(args.at)]
    recs = [orchestrator.run_cycle(t.to_pydatetime(), scenario=args.week) for t in times]

    if args.week:
        changes = transitions(recs)
        print(
            f"Неделя «{args.week}» с {rd.DEMO_WEEKS[args.week]:%Y-%m-%d %H:%M}: "
            f"{len(recs)} решений раз в {args.every} мин, из них смен решения {len(changes)}"
        )
        for rec in changes:
            print("  " + one_line(rec))
        counts = pd.Series([outcome(r) for r in recs]).value_counts().to_dict()
        share = {k: f"{v / len(recs):.0%}" for k, v in counts.items()}
        print(f"  итог: {counts} — {share}")
        print()
    shown = recs if args.all_cards else [next((r for r in recs if outcome(r) == "RECOMMEND"), recs[0])]
    for rec in shown:
        print(full_card(rec))
        print()
    print(f"JSON-трассы: {rd.rel(orchestrator.artifacts_dir)}")
    return recs


if __name__ == "__main__":
    main()
