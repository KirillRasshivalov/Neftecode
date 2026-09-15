from __future__ import annotations

from itertools import product
from typing import Any

from neftecode.agents.base import QualityAgent, ReliabilityAgent
from neftecode.data.config import load_constraints, load_tags_whitelist
from neftecode.domain.actions import ControlAction
from neftecode.domain.agent_results import ScoredScenario
from neftecode.domain.state import ProcessState
from neftecode.safety.constraints import HardConstraints

#: Metrics where a larger value is better. The score is lower-is-better, so these
#: are subtracted; everything else (risks, energy) is added.
_BENEFIT_METRICS = frozenset({"throughput"})


class OptimizerAgent:
    def __init__(
        self,
        quality_agent: QualityAgent,
        reliability_agent: ReliabilityAgent,
        constraints: HardConstraints | None = None,
        whitelist: dict[str, Any] | None = None,
        ranking_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.quality_agent = quality_agent
        self.reliability_agent = reliability_agent
        self.constraints = constraints or HardConstraints(load_constraints())
        self.whitelist = whitelist or load_tags_whitelist()
        cfg = ranking_cfg if ranking_cfg is not None else load_constraints()
        self.weights = cfg.get("ranking_weights", {})

    def propose(self, state: ProcessState) -> list[ScoredScenario]:
        candidates = self._generate_candidates(state)
        scored: list[ScoredScenario] = []

        for action in candidates:
            quality = self.quality_agent.assess(state, action)
            reliability = self.reliability_agent.assess(state, action)
            ok, reasons = self.constraints.check_action(state, action, quality, reliability)
            metrics = {
                "quality_risk": quality.risk_of_spec_breach,
                "equipment_risk": reliability.risk_index,
                "throughput": 0.0,
                "energy_or_cost_proxy": 0.0,
            }
            score = self._score(metrics) if ok else None
            scored.append(
                ScoredScenario(
                    action=action,
                    quality=quality,
                    reliability=reliability,
                    feasible=ok,
                    rejection_reasons=reasons,
                    score=score,
                    metrics=metrics,
                )
            )

        feasible = [s for s in scored if s.feasible]
        # Lower score is better. On a tie, holding the current regime wins: a change
        # that buys nothing is not worth an operator action.
        feasible.sort(
            key=lambda s: (s.score if s.score is not None else float("inf"), not s.action.is_noop())
        )
        return feasible

    def _generate_candidates(self, state: ProcessState) -> list[ControlAction]:
        params = self.whitelist.get("controllable_parameters", [])
        tag_options: list[list[tuple[str, float]]] = []

        for item in params:
            tag = item["tag"]
            current = state.controllable.get(tag)
            if current is None:
                continue
            deltas = item.get("deltas") or [0.0]
            options = [(tag, current + float(d)) for d in deltas]
            tag_options.append(options)

        if not tag_options:
            return [ControlAction(changes={}, label="noop")]

        actions: list[ControlAction] = []
        for combo in product(*tag_options):
            changes = {tag: value for tag, value in combo}
            delta_only = {
                tag: val
                for tag, val in changes.items()
                if state.controllable.get(tag) is None or abs(val - state.controllable[tag]) > 1e-9
            }
            label = "noop" if not delta_only else "delta:" + ",".join(delta_only.keys())
            actions.append(ControlAction(changes=changes if delta_only else {}, label=label))

        uniq: dict[str, ControlAction] = {}
        for action in actions:
            key = str(sorted(action.changes.items()))
            uniq[key] = action
        return list(uniq.values())

    def _score(self, metrics: dict[str, float]) -> float:
        total = 0.0
        for key, weight in self.weights.items():
            sign = -1.0 if key in _BENEFIT_METRICS else 1.0
            total += sign * float(weight) * float(metrics.get(key, 0.0))
        return total
