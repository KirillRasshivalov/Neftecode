from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from neftecode.agents.base import QualityAgent, ReliabilityAgent
from neftecode.agents.optimizer import OptimizerAgent
from neftecode.agents.quality import QualityAgentStub
from neftecode.agents.reliability import ReliabilityAgentStub
from neftecode.data.config import load_constraints, load_tags_whitelist, project_root
from neftecode.data.state_builder import ProcessStateBuilder
from neftecode.domain.agent_results import QualityAssessment, ReliabilityAssessment, ScoredScenario
from neftecode.domain.recommendation import OperatorRecommendation
from neftecode.domain.state import ProcessState
from neftecode.domain.actions import ControlAction
from neftecode.orchestration.explain import build_explanation, describe_blend, describe_budget, describe_economy
from neftecode.safety.constraints import HardConstraints
from neftecode.safety.data_quality_gate import DataQualityGate

RECOMMEND = "recommend"
HOLD = "hold"
REFUSE = "refuse"

#: Used when configs/constraints.yaml has no `decision` section.
DEFAULT_HOLD_MARGIN = 5.0
DEFAULT_ACT_WHEN_BREACH_RISK_AT_LEAST = 0.5
#: The classifier lives on a different scale, so it gets its own threshold. 0.5 would
#: never fire at a 15 % base rate; 0.235 catches half the breaches on the training
#: period (`models/breach_classifier.json`).
DEFAULT_ACT_WHEN_BREACH_RISK_AT_LEAST_CLASSIFIER = 0.2512

#: How the quality agent arrived at `risk_of_spec_breach`, and what to call it.
RISK_SOURCE_LABELS = {"classifier": "классификатор", "interval": "оценка по интервалу"}

#: Trigger kinds. `reliability` alone is answered differently from the rest: raising the
#: reactor temperature lowers sulfur and so scores well, but answering "the equipment is
#: working hard" by making it work harder is not an answer. When severity is the only
#: complaint, only steps that do not raise it are considered.
QUALITY_TRIGGERS = frozenset({"lab_off_spec", "breach_risk", "constraint"})

#: The energy lever: a cooler reactor spends less and ages the catalyst slower.
ECONOMY_LEVER = "242000:T5"
ECONOMY_MAX_STEPS = 10


class Orchestrator:
    """One decision cycle: state, data checks, candidates, hard filter, decision.

    There are three outcomes. Until the contract gains an explicit `outcome` field they
    are expressed with the existing fields, and recorded in `audit["decision"]`:

    | outcome   | refuse | proposed_action | refuse_reason |
    |-----------|--------|-----------------|---------------|
    | recommend | False  | the action      | None          |
    | hold      | False  | None            | None          |
    | refuse    | True   | None            | set           |

    A change is recommended only when there is a reason to act **and** the change is
    worth it:

    - a trigger exists: the latest lab result is off-spec, the predicted breach
      probability is high, operating severity is elevated, or the current regime fails
      a hard constraint. The probability threshold depends on which estimate produced
      it — a classifier probability and an interval probability are not comparable, so
      each has its own value in `decision`;
    - the best feasible change beats holding by at least `hold_margin` score units, or
      holding itself is infeasible.

    Otherwise the regime is held. A regime that is within spec is not changed just
    because a change would look slightly better on paper.
    """

    def __init__(
        self,
        quality_agent: QualityAgent | None = None,
        reliability_agent: ReliabilityAgent | None = None,
        state_builder: ProcessStateBuilder | None = None,
        artifacts_dir: Path | None = None,
        whitelist: dict[str, Any] | None = None,
        constraints_cfg: dict[str, Any] | None = None,
        blending_agent: Any | None = None,
    ) -> None:
        self.quality_agent = quality_agent or QualityAgentStub()
        self.reliability_agent = reliability_agent or ReliabilityAgentStub()
        self.state_builder = state_builder or ProcessStateBuilder()
        # Both are injectable so a caller can run the cycle on real levers without
        # editing the YAML. With no arguments the behaviour is unchanged.
        self.constraints_cfg = constraints_cfg if constraints_cfg is not None else load_constraints()
        self.whitelist = whitelist if whitelist is not None else load_tags_whitelist()
        self.gate = DataQualityGate(self.constraints_cfg)
        self.hard = HardConstraints(self.constraints_cfg, whitelist=self.whitelist)
        self.optimizer = OptimizerAgent(
            quality_agent=self.quality_agent,
            reliability_agent=self.reliability_agent,
            constraints=self.hard,
            whitelist=self.whitelist,
            ranking_cfg=self.constraints_cfg,
        )
        decision = self.constraints_cfg.get("decision") or {}
        self.hold_margin = float(decision.get("hold_margin", DEFAULT_HOLD_MARGIN))
        self.act_risk = float(decision.get("act_when_breach_risk_at_least", DEFAULT_ACT_WHEN_BREACH_RISK_AT_LEAST))
        self.act_risk_classifier = float(decision.get(
            "act_when_breach_risk_at_least_classifier",
            DEFAULT_ACT_WHEN_BREACH_RISK_AT_LEAST_CLASSIFIER,
        ))
        self.sulfur_limit = float((self.constraints_cfg.get("hard") or {}).get("sulfur_mg_kg_max", 10.0))
        # Optional: without it the card has no blend block and nothing else changes.
        self.blending_agent = blending_agent
        self.artifacts_dir = artifacts_dir or (project_root() / "artifacts")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ cycle

    def run_cycle(self, timestamp: datetime, scenario: str | None = None) -> OperatorRecommendation:
        state = self.state_builder.build(timestamp, scenario=scenario)
        gate = self.gate.check(state)
        baseline_q = self.quality_agent.assess(state)
        baseline_r = self.reliability_agent.assess(state)
        audit: dict[str, Any] = {
            "state": state.model_dump(mode="json"),
            "baseline_quality": baseline_q.model_dump(mode="json"),
            "baseline_reliability": baseline_r.model_dump(mode="json"),
        }

        if state.data_flags.get("running") is False:
            msg = self._message(
                "unit_not_running",
                "Надёжной рекомендации нет: установка не работает — режимные изменения не рассматриваются.",
            )
            detail = state.data_flags.get("running_detail")
            audit["decision"] = {"outcome": REFUSE, "why": "установка не работает", "triggers": []}
            return self._refuse(
                timestamp, state, f"{msg} Детали: {detail}" if detail else msg,
                problem="установка не работает", confidence=0.0, audit=audit,
            )

        if not gate.ok:
            msg = self._message("incomplete_data", "Надёжной рекомендации нет: входные данные неполны или несогласованы.")
            if any("age" in r.lower() or "LIMS" in r or "PAK" in r for r in gate.reasons):
                msg = self._message("stale_lims", msg)
            audit["gate"] = gate.reasons
            audit["decision"] = {"outcome": REFUSE, "why": "проверка данных не пройдена", "triggers": []}
            return self._refuse(
                timestamp, state, f"{msg} Детали: {'; '.join(gate.reasons)}",
                problem="; ".join(gate.reasons), confidence=0.0, audit=audit,
            )

        # The blending agent speaks first. How much sulfur the tank can take sets the
        # limit the hydrotreating regime is checked against — the organisers' 10 mg/kg
        # is a limit on the commercial product, not on this intermediate.
        budget = self._budget()
        limit = self._sulfur_limit_for(budget)
        if budget is not None:
            audit["sulfur_budget"] = {**budget.model_dump(mode="json"), "limit_in_force_mg_kg": limit}
        optimizer, hard = self._planners(limit)
        scenarios = optimizer.evaluate(state)
        feasible = [s for s in scenarios if s.feasible]
        hold = next((s for s in scenarios if s.action.is_noop()), None)
        found = self._triggers(state, hold, baseline_r)
        kinds = [kind for kind, _ in found]
        triggers = [text for _, text in found]
        audit["n_candidates"] = len(scenarios)
        audit["n_feasible"] = len(feasible)
        audit["trigger_kinds"] = kinds

        # Severity on its own must not be answered by a step that raises severity.
        equipment_only = bool(kinds) and not QUALITY_TRIGGERS.intersection(kinds)
        if equipment_only:
            kept = self._not_harder_on_equipment(feasible, hold)
            audit["equipment_only_filter"] = {
                "applied": True,
                "dropped": len(feasible) - len(kept),
                "hold_risk_index": None if hold is None else hold.reliability.risk_index,
            }
            feasible = kept

        if not feasible:
            msg = self._message("no_feasible", "Надёжной рекомендации нет: нет допустимых вариантов.")
            detail = "; ".join(self._rejection_detail(hold)[:3]) if hold else ""
            text = (
                f"{msg} Текущий режим не проходит ограничения, и ни один допустимый шаг этого "
                f"не исправляет — нужно решение человека."
                + (f" Детали: {detail}" if detail else "")
            )
            audit["decision"] = {"outcome": REFUSE, "why": "допустимых вариантов нет", "triggers": triggers}
            # The regime cannot be fixed in one step, but the commercial tank may still be:
            # say what the blend can do with what the unit is making now.
            return self._refuse(
                timestamp, state, text,
                problem=self._problem(triggers, baseline_q, baseline_r),
                confidence=baseline_q.confidence, audit=audit,
                blend=self._blend(hold), hard=hard,
            )

        best = feasible[0]
        hold_feasible = hold is not None and hold.feasible
        improvement = (hold.score - best.score) if hold_feasible else None

        if best.action.is_noop():
            outcome, why = HOLD, "удержание режима оценено не хуже любого изменения"
        elif not hold_feasible:
            outcome, why = RECOMMEND, "текущий режим не проходит ограничения, выбран лучший допустимый шаг"
        elif not triggers:
            outcome, why = HOLD, "признаков проблемы нет — режим в пределах ограничений, менять его незачем"
        elif improvement < self.hold_margin:
            outcome, why = HOLD, (
                f"проблема есть, но лучший шаг улучшает оценку лишь на {improvement:.1f} "
                f"при пороге {self.hold_margin:g}"
            )
        else:
            outcome, why = RECOMMEND, (
                f"лучший из {len(feasible)} допустимых вариантов улучшает оценку на {improvement:.1f} "
                f"(порог {self.hold_margin:g})"
            )

        chosen = best if outcome == RECOMMEND else hold
        blend = self._blend(chosen)
        if blend is not None:
            audit["blend"] = blend
        # Room to cool the reactor, only when the regime is held with nothing wrong.
        # Computed before the recommendation is built: the model copies the audit.
        economy = self._economy(state, limit) if outcome == HOLD and not triggers else None
        if economy is not None:
            audit["economy"] = economy
        risk_threshold, risk_source = self._risk_threshold(hold) if hold else (self.act_risk, None)
        audit["decision"] = {
            "outcome": outcome,
            "why": why,
            "triggers": triggers,
            "hold_margin": self.hold_margin,
            "breach_risk_source": risk_source,
            "breach_risk_threshold": risk_threshold,
            "improvement_over_hold": None if improvement is None else round(improvement, 4),
            "best_candidate": best.action.label,
        }
        rec = OperatorRecommendation(
            timestamp=timestamp,
            refuse=False,
            problem_or_risk=self._problem(triggers, baseline_q, baseline_r),
            proposed_action=chosen.action if outcome == RECOMMEND else None,
            expected_effect={
                "quality": chosen.quality.model_dump(mode="json"),
                "reliability": chosen.reliability.model_dump(mode="json"),
                "score": chosen.score,
                "hold_score": hold.score if hold_feasible else None,
                "blend": blend,
            },
            constraints_checked=self._checked(blend, hard),
            confidence=chosen.quality.confidence,
            alternatives=[s for s in feasible if s is not chosen][:2],
            audit=audit,
        )
        rec.explanation = build_explanation(
            state, chosen, refuse=False, refuse_reason=None,
            outcome=outcome, why=why, triggers=triggers, hold=hold,
        )
        if blend is not None:
            rec.explanation = f"{rec.explanation} {describe_blend(blend)}"
        budget_text = describe_budget(
            audit.get("sulfur_budget"), chosen.quality.metrics.get("sulfur_mg_kg"), self.sulfur_limit
        )
        if budget_text:
            rec.explanation = f"{rec.explanation} {budget_text}"
        if economy is not None:
            rec.explanation = f"{rec.explanation} {describe_economy(economy)}"
        self._persist(rec)
        return rec

    # ---------------------------------------------------------------- helpers

    def _triggers(
        self,
        state: ProcessState,
        hold: ScoredScenario | None,
        reliability: ReliabilityAssessment,
    ) -> list[tuple[str, str]]:
        """Evidence that something needs a response, as (kind, text). No trigger, no change."""
        triggers: list[tuple[str, str]] = []
        lab = state.quality.get("sulfur_mg_kg")
        if lab is not None and lab.value is not None and lab.source == "lims" and lab.value > self.sulfur_limit:
            triggers.append((
                "lab_off_spec",
                f"последний анализ ЛИМС вне спецификации: {lab.value:.1f} > {self.sulfur_limit:g} мг/кг",
            ))
        if hold is not None:
            threshold, source = self._risk_threshold(hold)
            if hold.quality.risk_of_spec_breach >= threshold:
                triggers.append((
                    "breach_risk",
                    f"вероятность нарушения спецификации {hold.quality.risk_of_spec_breach:.0%} "
                    f"(порог {threshold:.0%}, {RISK_SOURCE_LABELS.get(source, source)})",
                ))
        if reliability.risk_class in ("medium", "high"):
            triggers.append((
                "reliability",
                f"тяжесть режима {reliability.risk_class} ({reliability.risk_index:.2f})",
            ))
        if hold is not None and not hold.feasible:
            triggers.append((
                "constraint",
                "текущий режим не проходит ограничения: " + "; ".join(self._rejection_detail(hold)[:2]),
            ))
        return triggers

    @staticmethod
    def _not_harder_on_equipment(
        feasible: list[ScoredScenario], hold: ScoredScenario | None
    ) -> list[ScoredScenario]:
        """Keep holding, plus only the steps that do not push severity above holding."""
        if hold is None:
            return feasible
        limit = hold.reliability.risk_index + 1e-9
        return [s for s in feasible if s.action.is_noop() or s.reliability.risk_index <= limit]

    @staticmethod
    def _rejection_detail(scenario: ScoredScenario) -> list[str]:
        """Why a candidate failed, in the words of the agent that knows.

        The hard check reports a mode the reliability agent rejected only as "not
        allowed"; the operator needs which lever, where it is and what the range is, and
        that is in the reliability agent's own risk factors.
        """
        detail = [r for r in scenario.rejection_reasons if "reliability agent" not in r]
        if not scenario.reliability.is_mode_allowed:
            detail = [
                f for f in scenario.reliability.risk_factors
                if "диапазон" in f or "Индекс тяжести" in f or "не работает" in f
            ] + detail
        return detail or list(scenario.rejection_reasons)

    def _risk_threshold(self, hold: ScoredScenario) -> tuple[float, str]:
        """The threshold that applies to this probability, and where it came from."""
        source = (hold.quality.details or {}).get("risk_source") or "interval"
        threshold = self.act_risk_classifier if source == "classifier" else self.act_risk
        return threshold, source

    @staticmethod
    def _problem(triggers: list[str], quality: QualityAssessment, reliability: ReliabilityAssessment) -> str:
        if triggers:
            return "; ".join(triggers)
        sulfur = quality.metrics.get("sulfur_mg_kg")
        text = "признаков проблемы нет"
        if sulfur is not None:
            text += f": прогноз серы {sulfur:.2f} мг/кг"
        return f"{text}, тяжесть режима {reliability.risk_class}"

    def _blend(self, scenario: ScoredScenario | None) -> dict | None:
        """The commercial blend for this regime, if a blending agent is attached."""
        if self.blending_agent is None or scenario is None:
            return None
        sulfur = scenario.quality.metrics.get("sulfur_mg_kg")
        return self.blending_agent.blend(sulfur).model_dump(mode="json")

    def _budget(self):
        """The tank's sulfur budget, when a blending agent is attached."""
        if self.blending_agent is None or not hasattr(self.blending_agent, "sulfur_budget"):
            return None
        return self.blending_agent.sulfur_budget()

    def _sulfur_limit_for(self, budget) -> float:
        """The budget may relax the product limit, never tighten it.

        Without a blending agent the regime was already held to that limit; the budget
        only adds the room the tank provides.
        """
        if budget is None or budget.budget_mg_kg is None:
            return self.sulfur_limit
        return max(self.sulfur_limit, float(budget.budget_mg_kg))

    def _planners(self, limit: float) -> tuple[OptimizerAgent, HardConstraints]:
        """The optimizer and the hard check for this cycle's sulfur limit.

        The hard check itself is unchanged; it simply receives the limit in force.
        """
        if abs(limit - self.sulfur_limit) < 1e-9:
            return self.optimizer, self.hard
        cfg = copy.deepcopy(self.constraints_cfg)
        cfg.setdefault("hard", {})["sulfur_mg_kg_max"] = round(limit, 3)
        hard = HardConstraints(cfg, whitelist=self.whitelist)
        optimizer = OptimizerAgent(
            quality_agent=self.quality_agent,
            reliability_agent=self.reliability_agent,
            constraints=hard,
            whitelist=self.whitelist,
            ranking_cfg=cfg,
        )
        return optimizer, hard

    def _economy(self, state: ProcessState, limit: float) -> dict | None:
        """How far the reactor could cool with the diesel still inside the sulfur limit.

        Information for the technologist, not a recommendation: the regime is on
        specification, and a cooler reactor spends less energy and ages the catalyst
        slower — the organisers: the more sulfur the hydrotreated diesel may keep, the
        cheaper it is to make. The response is the quality agent's what-if, an
        assumption, and the card says so.
        """
        item = next(
            (p for p in self.whitelist.get("controllable_parameters", []) if p.get("tag") == ECONOMY_LEVER),
            None,
        )
        current = state.controllable.get(ECONOMY_LEVER)
        if item is None or current is None:
            return None
        step = max((abs(float(d)) for d in item.get("deltas", []) if d), default=0.0)
        lowest = float((item.get("range") or {}).get("min", float("-inf")))
        if step <= 0.0:
            return None
        found = None
        for k in range(1, ECONOMY_MAX_STEPS + 1):
            target = float(current) - k * step
            if target < lowest:
                break
            quality = self.quality_agent.assess(state, ControlAction(changes={ECONOMY_LEVER: round(target, 4)}))
            sulfur = quality.metrics.get("sulfur_mg_kg")
            if sulfur is None or sulfur > limit:
                break
            found = {
                "lever": ECONOMY_LEVER,
                "from": round(float(current), 3),
                "to": round(target, 3),
                "delta": round(-k * step, 3),
                "sulfur_mg_kg": round(float(sulfur), 3),
                "limit_mg_kg": round(limit, 3),
                "needs_blend": float(sulfur) > self.sulfur_limit,
            }
        return found

    def _checked(self, blend: dict | None, hard: HardConstraints | None = None) -> list[str]:
        labels = list((hard or self.hard).checked_labels())
        if blend is not None and self.blending_agent is not None:
            labels += self.blending_agent.labels()
        return labels

    def _message(self, key: str, default: str) -> str:
        return (self.constraints_cfg.get("refuse_messages") or {}).get(key, default)

    def _refuse(
        self,
        timestamp: datetime,
        state: ProcessState,
        reason: str,
        *,
        problem: str,
        confidence: float,
        audit: dict[str, Any],
        blend: dict | None = None,
        hard: HardConstraints | None = None,
    ) -> OperatorRecommendation:
        if blend is not None:
            audit["blend"] = blend
        rec = OperatorRecommendation(
            timestamp=timestamp,
            refuse=True,
            refuse_reason=reason,
            problem_or_risk=problem,
            expected_effect={"blend": blend} if blend is not None else {},
            constraints_checked=self._checked(blend, hard),
            confidence=confidence,
            audit=audit,
        )
        rec.explanation = build_explanation(state, None, refuse=True, refuse_reason=reason)
        if blend is not None:
            rec.explanation = f"{rec.explanation} {describe_blend(blend)}"
        self._persist(rec)
        return rec

    def _persist(self, rec: OperatorRecommendation) -> Path:
        stamp = rec.timestamp.strftime("%Y%m%dT%H%M%S")
        path = self.artifacts_dir / f"recommendation_{stamp}.json"
        path.write_text(json.dumps(rec.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
        return path
