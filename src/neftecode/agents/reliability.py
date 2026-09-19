from __future__ import annotations

from neftecode.domain.actions import ControlAction
from neftecode.domain.agent_results import ReliabilityAssessment
from neftecode.domain.state import ProcessState


class ReliabilityAgentStub:
    def assess(
        self,
        state: ProcessState,
        action: ControlAction | None = None,
    ) -> ReliabilityAssessment:
        factors: list[str] = []
        risk = 0.2

        for tag, value in (action.changes if action else state.controllable).items():
            if value is None:
                continue
            if "TEMP" in tag.upper() and (value < 290 or value > 370):
                risk += 0.3
                factors.append(f"{tag} near model boundary ({value})")

        risk = min(1.0, risk)
        allowed = risk < 0.85
        return ReliabilityAssessment(
            risk_index=risk,
            risk_class="high" if risk >= 0.7 else "medium" if risk >= 0.4 else "low",
            risk_factors=factors,
            is_mode_allowed=allowed,
            soft_constraints=[] if allowed else ["reduce severity before optimizing throughput"],
            assumptions=["ReliabilityAgentStub uses proxy thresholds without labeled failures."],
        )


class ReliabilityAgentML:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path

    def assess(
        self,
        state: ProcessState,
        action: ControlAction | None = None,
    ) -> ReliabilityAssessment:
        raise NotImplementedError("ML-2: implement reliability/severity model here")


class ReliabilityAgentBaseline:
    """Operating-severity proxy on reference-resolved levers only (CLAUDE.md §8.3).

    The package has no labelled failures, no catalyst age and no bed temperatures, so
    severity is a transparent proxy with three parts:

    - temperature: how much hotter `242000:T5` runs than the training-period median at
      the same feed rate `242000:F26`; rising reaction temperature at constant
      throughput is the standard catalyst-deactivation signature;
    - throughput: feed above its typical value;
    - step: how large the proposed change is.

    Reference statistics come from `scripts/analysis/reliability_reference.py`
    (training period only) and are passed in: the agent reads no files
    (CLAUDE.md §2 rule 10). Weights and thresholds are assumptions.
    """

    W_TEMPERATURE = 0.5
    W_THROUGHPUT = 0.3
    W_STEP = 0.2
    MEDIUM_AT = 0.4
    HIGH_AT = 0.7
    NOT_ALLOWED_AT = 0.85
    Z_FULL = 3.0            # T5 this many sigma above typical: full temperature component
    THROUGHPUT_FULL = 0.20  # feed 20 % above typical: full throughput component
    STEPS_FULL = 2.0        # a lever moved by two steps at once: full step component

    T5 = "242000:T5"
    F26 = "242000:F26"

    def __init__(self, reference: dict) -> None:
        missing = [key for key in ("envelope", "t5_by_f26", "typical_f26", "steps") if key not in reference]
        if missing:
            raise ValueError(f"reliability reference is missing {missing}")
        self.reference = reference

    def assess(
        self,
        state: ProcessState,
        action: ControlAction | None = None,
    ) -> ReliabilityAssessment:
        if state.data_flags.get("running") is False:
            return ReliabilityAssessment(
                risk_index=1.0,
                risk_class="high",
                risk_factors=["Установка не работает: расход сырья ниже рабочего порога."],
                is_mode_allowed=False,
                soft_constraints=["не выдавать режимных рекомендаций до пуска установки"],
                assumptions=self._assumptions(),
                details={"running": False},
            )

        values = dict(state.controllable)
        if action is not None and action.changes:
            values.update({tag: float(v) for tag, v in action.changes.items() if v is not None})

        factors: list[str] = []
        soft: list[str] = []
        allowed = True
        for tag, bounds in self.reference["envelope"].items():
            value = values.get(tag)
            if value is None:
                continue
            if value < bounds["p01"] or value > bounds["p99"]:
                allowed = False
                factors.append(
                    f"{tag} = {value:.4g} вне рабочего диапазона обучающего периода "
                    f"[{bounds['p01']:.4g}; {bounds['p99']:.4g}]."
                )

        temperature, z = self._temperature(values, factors)
        throughput, feed_ratio = self._throughput(values, factors)
        step, max_steps = self._step(state, action)

        risk = self.W_TEMPERATURE * temperature + self.W_THROUGHPUT * throughput + self.W_STEP * step
        if temperature >= 0.5:
            soft.append(f"не повышать {self.T5}")
        if throughput >= 0.5:
            soft.append(f"не повышать {self.F26}")
        if risk >= self.NOT_ALLOWED_AT:
            allowed = False
            factors.append(f"Индекс тяжести режима {risk:.2f} выше порога {self.NOT_ALLOWED_AT}.")

        return ReliabilityAssessment(
            risk_index=round(risk, 4),
            risk_class="high" if risk >= self.HIGH_AT else "medium" if risk >= self.MEDIUM_AT else "low",
            risk_factors=factors,
            is_mode_allowed=allowed,
            soft_constraints=soft,
            assumptions=self._assumptions(),
            details={
                "components": {
                    "temperature": round(temperature, 4),
                    "throughput": round(throughput, 4),
                    "step": round(step, 4),
                },
                "t5_z": None if z is None else round(z, 3),
                "feed_to_typical": None if feed_ratio is None else round(feed_ratio, 4),
                "max_steps": round(max_steps, 4),
                "running": state.data_flags.get("running"),
            },
        )

    def _temperature(self, values: dict[str, float], factors: list[str]) -> tuple[float, float | None]:
        t5, f26 = values.get(self.T5), values.get(self.F26)
        if t5 is None or f26 is None:
            return 0.0, None
        band = next((b for b in self.reference["t5_by_f26"] if b["lo"] <= f26 < b["hi"]), None)
        if band is None:
            factors.append(f"Нет опорной статистики {self.T5} при {self.F26} = {f26:.4g}.")
            return 0.0, None
        z = (t5 - band["median"]) / band["sigma"]
        if z >= 2.0:
            factors.append(
                f"{self.T5} на {z:.1f}σ выше типичного при таком расходе — возможная дезактивация катализатора."
            )
        return min(max(z / self.Z_FULL, 0.0), 1.0), z

    def _throughput(self, values: dict[str, float], factors: list[str]) -> tuple[float, float | None]:
        f26 = values.get(self.F26)
        typical = float(self.reference["typical_f26"])
        if f26 is None or typical <= 0:
            return 0.0, None
        ratio = f26 / typical
        if ratio > 1.1:
            factors.append(f"{self.F26} на {100 * (ratio - 1):.0f} % выше типичного расхода сырья.")
        return min(max((ratio - 1.0) / self.THROUGHPUT_FULL, 0.0), 1.0), ratio

    def _step(self, state: ProcessState, action: ControlAction | None) -> tuple[float, float]:
        if action is None or not action.changes:
            return 0.0, 0.0
        steps = self.reference["steps"]
        largest = 0.0
        for tag, new in action.changes.items():
            current, size = state.controllable.get(tag), steps.get(tag)
            if current is None or new is None or not size:
                continue
            largest = max(largest, abs(float(new) - float(current)) / float(size))
        return min(largest / self.STEPS_FULL, 1.0), largest

    @staticmethod
    def _assumptions() -> list[str]:
        return [
            "Тяжесть режима — прокси: в пакете нет данных об отказах, возрасте катализатора и температурах по слоям.",
            "Прокси дезактивации: насколько 242000:T5 выше медианы обучающего периода при том же 242000:F26.",
            "Рабочий диапазон — p01–p99 обучающего периода на рабочих строках; модельная граница, не заводской лимит.",
            "Веса компонентов (0.5 / 0.3 / 0.2) и пороги классов выбраны, а не обучены.",
        ]
