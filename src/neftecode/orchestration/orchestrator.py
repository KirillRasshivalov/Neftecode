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
from neftecode.domain.recommendation import OperatorRecommendation
from neftecode.orchestration.explain import build_explanation
from neftecode.safety.constraints import HardConstraints
from neftecode.safety.data_quality_gate import DataQualityGate


class Orchestrator:
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
        self.artifacts_dir = artifacts_dir or (project_root() / "artifacts")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def run_cycle(self, timestamp: datetime, scenario: str | None = None) -> OperatorRecommendation:
        state = self.state_builder.build(timestamp, scenario=scenario)
        gate = self.gate.check(state)

        baseline_q = self.quality_agent.assess(state)
        baseline_r = self.reliability_agent.assess(state)

        if state.data_flags.get("running") is False:
            rec = self._refuse_not_running(timestamp, state, baseline_q, baseline_r)
            self._persist(rec)
            return rec

        if not gate.ok:
            msg = self.constraints_cfg.get("refuse_messages", {}).get(
                "incomplete_data",
                "Надёжной рекомендации нет: входные данные неполны или несогласованы.",
            )
            if any("age" in r.lower() or "LIMS" in r or "PAK" in r for r in gate.reasons):
                msg = self.constraints_cfg.get("refuse_messages", {}).get("stale_lims", msg)
            rec = OperatorRecommendation(
                timestamp=timestamp,
                refuse=True,
                refuse_reason=f"{msg} Детали: {'; '.join(gate.reasons)}",
                problem_or_risk="; ".join(gate.reasons),
                constraints_checked=self.hard.checked_labels(),
                confidence=0.0,
                explanation=None,
                audit={
                    "state": state.model_dump(mode="json"),
                    "gate": gate.reasons,
                    "baseline_quality": baseline_q.model_dump(mode="json"),
                    "baseline_reliability": baseline_r.model_dump(mode="json"),
                },
            )
            rec.explanation = build_explanation(state, None, refuse=True, refuse_reason=rec.refuse_reason)
            self._persist(rec)
            return rec

        feasible = self.optimizer.propose(state)
        if not feasible:
            msg = self.constraints_cfg.get("refuse_messages", {}).get(
                "no_feasible",
                "Надёжной рекомендации нет: нет допустимых вариантов.",
            )
            rec = OperatorRecommendation(
                timestamp=timestamp,
                refuse=True,
                refuse_reason=msg,
                problem_or_risk=self._problem_summary(baseline_q, baseline_r),
                constraints_checked=self.hard.checked_labels(),
                confidence=baseline_q.confidence,
                audit={
                    "state": state.model_dump(mode="json"),
                    "baseline_quality": baseline_q.model_dump(mode="json"),
                    "baseline_reliability": baseline_r.model_dump(mode="json"),
                },
            )
            rec.explanation = build_explanation(state, None, refuse=True, refuse_reason=rec.refuse_reason)
            self._persist(rec)
            return rec

        best = feasible[0]
        alternatives = feasible[1:3]
        rec = OperatorRecommendation(
            timestamp=timestamp,
            refuse=False,
            problem_or_risk=self._problem_summary(baseline_q, baseline_r),
            proposed_action=best.action,
            expected_effect={
                "quality": best.quality.model_dump(mode="json"),
                "reliability": best.reliability.model_dump(mode="json"),
                "score": best.score,
            },
            constraints_checked=self.hard.checked_labels(),
            confidence=best.quality.confidence,
            alternatives=alternatives,
            audit={
                "state": state.model_dump(mode="json"),
                "baseline_quality": baseline_q.model_dump(mode="json"),
                "baseline_reliability": baseline_r.model_dump(mode="json"),
                "n_feasible": len(feasible),
            },
        )
        rec.explanation = build_explanation(state, best, refuse=False, refuse_reason=None)
        self._persist(rec)
        return rec

    def _refuse_not_running(
        self,
        timestamp: datetime,
        state: Any,
        baseline_q: Any,
        baseline_r: Any,
    ) -> OperatorRecommendation:
        """The unit is shut down: no regime change is meaningful, whatever else the data say."""
        msg = self.constraints_cfg.get("refuse_messages", {}).get(
            "unit_not_running",
            "Надёжной рекомендации нет: установка не работает — режимные изменения не рассматриваются.",
        )
        detail = state.data_flags.get("running_detail")
        rec = OperatorRecommendation(
            timestamp=timestamp,
            refuse=True,
            refuse_reason=f"{msg} Детали: {detail}" if detail else msg,
            problem_or_risk="установка не работает",
            constraints_checked=self.hard.checked_labels(),
            confidence=0.0,
            audit={
                "state": state.model_dump(mode="json"),
                "baseline_quality": baseline_q.model_dump(mode="json"),
                "baseline_reliability": baseline_r.model_dump(mode="json"),
            },
        )
        rec.explanation = build_explanation(state, None, refuse=True, refuse_reason=rec.refuse_reason)
        return rec

    @staticmethod
    def _problem_summary(quality: Any, reliability: Any) -> str:
        parts = [
            f"quality_risk={quality.risk_of_spec_breach:.2f}",
            f"reliability={reliability.risk_class}({reliability.risk_index:.2f})",
        ]
        sulfur = quality.metrics.get("sulfur_mg_kg")
        if sulfur is not None:
            parts.insert(0, f"sulfur={sulfur}")
        return "; ".join(parts)

    def _persist(self, rec: OperatorRecommendation) -> Path:
        stamp = rec.timestamp.strftime("%Y%m%dT%H%M%S")
        path = self.artifacts_dir / f"recommendation_{stamp}.json"
        path.write_text(json.dumps(rec.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
        return path
