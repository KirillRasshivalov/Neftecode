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
from datetime import datetime

import pandas as pd

from neftecode.agents import QualityAgentBaseline, ReliabilityAgentBaseline
from neftecode.data.config import load_constraints
from neftecode.domain.recommendation import OperatorRecommendation
from neftecode.domain.state import ProcessState, QualityReading, TagValue
from neftecode.orchestration.orchestrator import Orchestrator
from scripts import realdata as rd
from scripts.analysis.quality_baseline_fit import ewma_after, usable_labels

#: A daily lab result sampled around 10:00 becomes usable at 14:00.
DECISION_HOUR = 16


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

    def __init__(self, quality_params: dict) -> None:
        self.tele = rd.load_telemetry()
        self.labels = pd.read_parquet(rd.MODELS_DIR / "labels.parquet").sort_values("sampled_at", kind="stable")
        self.labels = self.labels.reset_index(drop=True)
        self.pak = rd.load_pak_sulfur()

        usable = usable_labels(self.labels)
        self.usable_available = usable["available_at"].reset_index(drop=True)
        self.usable_ewma = ewma_after(
            usable["sulfur_mg_kg"].to_numpy(),
            float(quality_params["alpha"]),
            float(quality_params["sulfur_median_train"]),
        )

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
                "scenario": scenario,
            },
            notes=[
                "Мост scripts/run_real.py: заглушки удалены, теги с префиксом установки, "
                "ЛИМС доступен через 4 ч после отбора.",
            ],
        )


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


def one_line(rec: OperatorRecommendation) -> str:
    state = rec.audit.get("state", {})
    flags = state.get("data_flags", {})
    lab = state.get("quality", {}).get("sulfur_mg_kg")
    base = _interval(rec.audit.get("baseline_quality"))
    lab_text = f"ЛИМС {lab['value']:.1f} ({lab['age_minutes'] / 60:.0f} ч)" if lab else "ЛИМС —"
    p95 = f"p95 {base['p95']:.1f}" if base else "p95 —"
    action = ""
    if outcome(rec) == "RECOMMEND":
        current = state.get("controllable", {})
        moved = {t: v for t, v in rec.proposed_action.changes.items() if abs(v - current.get(t, v)) > 1e-9}
        action = "  " + ", ".join(f"{t} {current[t]:.4g}→{v:.4g}" for t, v in moved.items())
    running = {True: "работает", False: "стоит", None: "?"}[flags.get("running")]
    return f"{rec.timestamp:%Y-%m-%d %H:%M}  {outcome(rec):<9}  {lab_text:<18} {p95:<9} установка {running}{action}"


def full_card(rec: OperatorRecommendation) -> str:
    state = rec.audit.get("state", {})
    flags = state.get("data_flags", {})
    current = state.get("controllable", {})
    lab = state.get("quality", {}).get("sulfur_mg_kg")
    pak = flags.get("pak_sulfur") or {}
    base = _interval(rec.audit.get("baseline_quality"))
    lines = [f"═══ {rec.timestamp:%Y-%m-%d %H:%M} · {outcome(rec)} ═══"]
    lines.append("1 Состояние   " + " · ".join(f"{t} {v:.5g}" for t, v in current.items()))
    lab_text = f"ЛИМС {lab['value']:.1f} мг/кг ({lab['age_minutes'] / 60:.0f} ч назад)" if lab else "ЛИМС нет"
    pak_text = (
        f"ПАК {pak['value']:.1f} ppm, {'исправен' if pak.get('healthy') else 'неисправен/заморожен'}"
        if pak else "ПАК нет"
    )
    ewma = flags.get("lab_sulfur_ewma")
    lines.append(f"              {lab_text} · сглаженное {ewma if ewma is not None else '—'} · {pak_text}")
    lines.append(f"2 Проблема    {rec.problem_or_risk}")
    if outcome(rec) == "RECOMMEND":
        moved = {t: v for t, v in rec.proposed_action.changes.items() if abs(v - current.get(t, v)) > 1e-9}
        lines.append("3 Действие    " + ", ".join(f"{t} {current[t]:.5g} → {v:.5g}" for t, v in moved.items()))
    elif outcome(rec) == "HOLD":
        lines.append("3 Действие    режим не менять")
    else:
        lines.append("3 Действие    —")
    if outcome(rec) != "REFUSE":
        best_q = rec.expected_effect.get("quality", {})
        best_r = rec.expected_effect.get("reliability", {})
        after = _interval(best_q)
        lines.append(
            f"4 Эффект      сера: среднее {after.get('mean', float('nan')):.2f}, p95 {after.get('p95', float('nan')):.2f}"
            f" (сейчас {base.get('mean', float('nan')):.2f} / {base.get('p95', float('nan')):.2f})"
            f" · риск спеки {best_q.get('risk_of_spec_breach', float('nan')):.2f}"
            f" · тяжесть {best_r.get('risk_index', float('nan')):.2f} ({best_r.get('risk_class', '?')})"
        )
    lines.append("5 Проверки    " + "; ".join(rec.constraints_checked))
    lines.append(f"6 Уверенность {rec.confidence}")
    lines.append(f"7 Почему      {rec.explanation}")
    if rec.refuse:
        lines.append(f"  Отказ       {rec.refuse_reason}")
    else:
        lines.append(f"  Допустимых вариантов: {rec.audit.get('n_feasible')}")
    return "\n".join(lines)


def decision_times(week: str) -> list[pd.Timestamp]:
    start = rd.DEMO_WEEKS[week]
    return [start.normalize() + pd.Timedelta(days=d, hours=DECISION_HOUR) for d in range(7)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="run_real", description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--week", choices=sorted(rd.DEMO_WEEKS), help="frozen demo week")
    group.add_argument("--at", help="ISO timestamp")
    parser.add_argument("--all-cards", action="store_true", help="print the full card for every decision")
    args = parser.parse_args(argv)

    quality_params = load_model_file("quality_baseline.json")
    reference = load_model_file("reliability_reference.json")
    orchestrator = Orchestrator(
        quality_agent=QualityAgentBaseline(quality_params),
        reliability_agent=ReliabilityAgentBaseline(reference),
        state_builder=RealStateBuilder(quality_params),
        artifacts_dir=rd.REPO_ROOT / "artifacts" / "real",
        whitelist=frozen_whitelist(reference),
        constraints_cfg=load_constraints(),
    )

    times = decision_times(args.week) if args.week else [pd.Timestamp(args.at)]
    recs = [orchestrator.run_cycle(t.to_pydatetime(), scenario=args.week) for t in times]

    if args.week:
        print(f"Неделя «{args.week}» с {rd.DEMO_WEEKS[args.week]:%Y-%m-%d %H:%M}, решение раз в сутки в {DECISION_HOUR}:00")
        for rec in recs:
            print("  " + one_line(rec))
        counts = pd.Series([outcome(r) for r in recs]).value_counts().to_dict()
        print(f"  итог: {counts}")
        print()
    shown = recs if args.all_cards else [next((r for r in recs if outcome(r) == "RECOMMEND"), recs[0])]
    for rec in shown:
        print(full_card(rec))
        print()
    print(f"JSON-трассы: {orchestrator.artifacts_dir}")


if __name__ == "__main__":
    main()
