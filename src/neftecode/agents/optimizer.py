from __future__ import annotations

from itertools import combinations, product
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

#: How many levers one recommendation may move, unless configured.
DEFAULT_MAX_LEVERS_PER_ACTION = 1


class OptimizerAgent:
    """Generate candidate actions, evaluate them, filter them, rank what is left.

    Each candidate is assessed by the quality and reliability agents and checked
    against the hard constraints. Infeasible candidates never get a score and are
    ranked after the feasible ones, so no weight can trade a violation away
    (CLAUDE.md §2 rule 4).

    Candidates move at most `max_levers_per_action` levers (one by default): an
    operator can carry out and verify one change at a time, and a card that moves four
    setpoints at once cannot say which of them did the work. Holding the current regime
    is always a candidate.
    """

    def __init__(
        self,
        quality_agent: QualityAgent,
        reliability_agent: ReliabilityAgent,
        constraints: HardConstraints | None = None,
        whitelist: dict[str, Any] | None = None,
        ranking_cfg: dict[str, Any] | None = None,
        max_levers_per_action: int | None = None,
    ) -> None:
        self.quality_agent = quality_agent
        self.reliability_agent = reliability_agent
        self.constraints = constraints or HardConstraints(load_constraints())
        self.whitelist = whitelist or load_tags_whitelist()
        cfg = ranking_cfg if ranking_cfg is not None else load_constraints()
        self.weights = cfg.get("ranking_weights", {})
        configured = (cfg.get("decision") or {}).get("max_levers_per_action")
        self.max_levers = int(max_levers_per_action or configured or DEFAULT_MAX_LEVERS_PER_ACTION)

    def evaluate(self, state: ProcessState) -> list[ScoredScenario]:
        """Every candidate: feasible ones ranked first, then the rejected ones.

        Lower score is better. On a tie, holding the current regime wins: a change
        that buys nothing is not worth an operator action.
        """
        scored: list[ScoredScenario] = []
        for action in self._generate_candidates(state):
            quality = self.quality_agent.assess(state, action)
            reliability = self.reliability_agent.assess(state, action)
            ok, reasons = self.constraints.check_action(state, action, quality, reliability)
            metrics = {
                "quality_risk": quality.risk_of_spec_breach,
                "equipment_risk": reliability.risk_index,
                "throughput": 0.0,
                "energy_or_cost_proxy": 0.0,
            }
            scored.append(
                ScoredScenario(
                    action=action,
                    quality=quality,
                    reliability=reliability,
                    feasible=ok,
                    rejection_reasons=reasons,
                    score=self._score(metrics) if ok else None,
                    metrics=metrics,
                )
            )

        feasible = sorted(
            (s for s in scored if s.feasible),
            key=lambda s: (s.score, not s.action.is_noop()),
        )
        rejected = [s for s in scored if not s.feasible]
        return feasible + rejected

    def propose(self, state: ProcessState) -> list[ScoredScenario]:
        """Feasible candidates only, best first."""
        return [s for s in self.evaluate(state) if s.feasible]

    def _generate_candidates(self, state: ProcessState) -> list[ControlAction]:
        """Hold, plus every combination of at most `max_levers` non-zero lever moves."""
        moves: dict[str, list[tuple[float, float]]] = {}
        for item in self.whitelist.get("controllable_parameters", []):
            tag = item["tag"]
            current = state.controllable.get(tag)
            if current is None:
                continue
            steps = [float(d) for d in (item.get("deltas") or []) if float(d) != 0.0]
            if steps:
                moves[tag] = [(d, current + d) for d in steps]

        actions = [ControlAction(changes={}, label="noop")]
        for size in range(1, self.max_levers + 1):
            for tags in combinations(moves, size):
                for combo in product(*(moves[tag] for tag in tags)):
                    changes = {tag: value for tag, (_, value) in zip(tags, combo)}
                    label = "delta:" + ",".join(f"{tag}{delta:+g}" for tag, (delta, _) in zip(tags, combo))
                    actions.append(ControlAction(changes=changes, label=label))
        return actions

    def _score(self, metrics: dict[str, float]) -> float:
        total = 0.0
        for key, weight in self.weights.items():
            sign = -1.0 if key in _BENEFIT_METRICS else 1.0
            total += sign * float(weight) * float(metrics.get(key, 0.0))
        return total
