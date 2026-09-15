from __future__ import annotations

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
from neftecode.orchestration.explain import build_explanation
from neftecode.safety.constraints import HardConstraints
from neftecode.safety.data_quality_gate import DataQualityGate

RECOMMEND = "recommend"
HOLD = "hold"
REFUSE = "refuse"

#: Used when configs/constraints.yaml has no `decision` section.
DEFAULT_HOLD_MARGIN = 5.0
DEFAULT_ACT_WHEN_BREACH_RISK_AT_LEAST = 0.5


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
      a hard constraint;
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
        self.sulfur_limit = float((self.constraints_cfg.get("hard") or {}).get("sulfur_mg_kg_max", 10.0))
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

        scenarios = self.optimizer.evaluate(state)
        feasible = [s for s in scenarios if s.feasible]
        hold = next((s for s in scenarios if s.action.is_noop()), None)
        triggers = self._triggers(state, hold, baseline_r)
        audit["n_candidates"] = len(scenarios)
        audit["n_feasible"] = len(feasible)

        if not feasible:
            msg = self._message("no_feasible", "Надёжной рекомендации нет: нет допустимых вариантов.")
            detail = "; ".join((hold.rejection_reasons if hold else [])[:3])
            text = (
                f"{msg} Текущий режим не проходит ограничения, и ни один допустимый шаг этого "
                f"не исправляет — нужно решение человека."
                + (f" Детали: {detail}" if detail else "")
            )
            audit["decision"] = {"outcome": REFUSE, "why": "допустимых вариантов нет", "triggers": triggers}
            return self._refuse(
                timestamp, state, text,
                problem=self._problem(triggers, baseline_q, baseline_r),
                confidence=baseline_q.confidence, audit=audit,
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
        audit["decision"] = {
            "outcome": outcome,
            "why": why,
            "triggers": triggers,
            "hold_margin": self.hold_margin,
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
            },
            constraints_checked=self.hard.checked_labels(),
            confidence=chosen.quality.confidence,
            alternatives=[s for s in feasible if s is not chosen][:2],
            audit=audit,
        )
        rec.explanation = build_explanation(
            state, chosen, refuse=False, refuse_reason=None,
            outcome=outcome, why=why, triggers=triggers, hold=hold,
        )
        self._persist(rec)
        return rec

    # ---------------------------------------------------------------- helpers

    def _triggers(
        self,
        state: ProcessState,
        hold: ScoredScenario | None,
        reliability: ReliabilityAssessment,
    ) -> list[str]:
        """Evidence that something needs a response. No trigger, no change."""
        triggers: list[str] = []
        lab = state.quality.get("sulfur_mg_kg")
        if lab is not None and lab.value is not None and lab.source == "lims" and lab.value > self.sulfur_limit:
            triggers.append(f"последний анализ ЛИМС вне спецификации: {lab.value:.1f} > {self.sulfur_limit:g} мг/кг")
        if hold is not None and hold.quality.risk_of_spec_breach >= self.act_risk:
            triggers.append(
                f"вероятность нарушения спецификации {hold.quality.risk_of_spec_breach:.0%} "
                f"(порог {self.act_risk:.0%})"
            )
        if reliability.risk_class in ("medium", "high"):
            triggers.append(f"тяжесть режима {reliability.risk_class} ({reliability.risk_index:.2f})")
        if hold is not None and not hold.feasible:
            triggers.append("текущий режим не проходит ограничения: " + "; ".join(hold.rejection_reasons[:2]))
        return triggers

    @staticmethod
    def _problem(triggers: list[str], quality: QualityAssessment, reliability: ReliabilityAssessment) -> str:
        if triggers:
            return "; ".join(triggers)
        sulfur = quality.metrics.get("sulfur_mg_kg")
        text = "признаков проблемы нет"
        if sulfur is not None:
            text += f": прогноз серы {sulfur:.2f} мг/кг"
        return f"{text}, тяжесть режима {reliability.risk_class}"

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
    ) -> OperatorRecommendation:
        rec = OperatorRecommendation(
            timestamp=timestamp,
            refuse=True,
            refuse_reason=reason,
            problem_or_risk=problem,
            constraints_checked=self.hard.checked_labels(),
            confidence=confidence,
            audit=audit,
        )
        rec.explanation = build_explanation(state, None, refuse=True, refuse_reason=reason)
        self._persist(rec)
        return rec

    def _persist(self, rec: OperatorRecommendation) -> Path:
        stamp = rec.timestamp.strftime("%Y%m%dT%H%M%S")
        path = self.artifacts_dir / f"recommendation_{stamp}.json"
        path.write_text(json.dumps(rec.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
        return path
