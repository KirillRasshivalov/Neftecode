from __future__ import annotations

import math

from neftecode.domain.actions import ControlAction
from neftecode.domain.agent_results import QualityAssessment
from neftecode.domain.state import ProcessState


class QualityAgentStub:
    def assess(
        self,
        state: ProcessState,
        action: ControlAction | None = None,
    ) -> QualityAssessment:
        sulfur_reading = state.quality.get("sulfur_mg_kg")
        sulfur = sulfur_reading.value if sulfur_reading else None

        if sulfur is not None and action and action.changes:
            delta_t = action.changes.get("PLACEHOLDER_HDT_TEMP")
            if delta_t is not None:
                current = state.controllable.get("PLACEHOLDER_HDT_TEMP")
                if current is not None:
                    sulfur = max(0.0, sulfur - 0.05 * (delta_t - current))

        risk = 0.0
        if sulfur is not None:
            risk = max(0.0, min(1.0, (sulfur - 8.0) / 4.0))

        return QualityAssessment(
            metrics={"sulfur_mg_kg": sulfur},
            risk_of_spec_breach=risk,
            confidence=0.4 if state.data_flags.get("scaffold_mode") else 0.6,
            horizon_minutes=60,
            features_used=sorted((action.changes if action else {}).keys()),
            assumptions=["QualityAgentStub uses heuristic sulfur response, not a trained VAK."],
        )


class QualityAgentML:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self._model = None

    def assess(
        self,
        state: ProcessState,
        action: ControlAction | None = None,
    ) -> QualityAssessment:
        raise NotImplementedError("ML-1: implement model load + feature mapping here")


class QualityAgentBaseline:
    """EWMA baseline: smoothed lab history, empirical interval, signed what-if.

    Works whether or not the telemetry predicts sulfur. Consecutive lab results are
    only weakly autocorrelated, so the prediction is an exponentially weighted mean of
    past results (smoothing factor chosen on the training period), not the last
    result. The interval is the training residual spread of that predictor, so the
    p95 is real even without a trained model. The as-of EWMA is computed by the state
    builder and arrives in `data_flags["lab_sulfur_ewma"]`; parameters are fitted by
    `scripts/analysis/quality_baseline_fit.py` and passed in. The agent reads no files
    (CLAUDE.md §2 rule 10).

    The what-if layer moves the prediction when a lever changes. Directions follow the
    physics and are fixed; magnitudes are assumptions until a model replaces them.
    """

    METRIC = "sulfur_mg_kg"
    SPEC_LIMIT_MG_KG = 10.0

    #: mg/kg of sulfur per unit change of a lever. Signs are physics, sizes are assumed.
    DEFAULT_SENSITIVITY: dict[str, float] = {
        "242000:T5": -0.35,     # hotter reactor: deeper desulfurisation
        "242000:F26": 0.05,     # more feed over the same catalyst: less contact time
        "242000:F15": 0.0005,   # more quench: cooler second bed
    }
    #: mg/kg per unit change of the gas-to-feed ratio 242000:F2 / 242000:F26.
    DEFAULT_RATIO_SENSITIVITY = -0.025

    BASE_CONFIDENCE = 0.8
    AGE_HALFLIFE_HOURS = 72.0

    def __init__(
        self,
        params: dict,
        sensitivity: dict[str, float] | None = None,
        ratio_sensitivity: float | None = None,
    ) -> None:
        required = ("alpha", "resid_q05", "resid_q95", "ref_gap_hours", "sulfur_median_train")
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(f"quality baseline params are missing {missing}")
        self.params = dict(params)
        self.sensitivity = dict(self.DEFAULT_SENSITIVITY if sensitivity is None else sensitivity)
        self.ratio_sensitivity = (
            self.DEFAULT_RATIO_SENSITIVITY if ratio_sensitivity is None else ratio_sensitivity
        )

    def assess(
        self,
        state: ProcessState,
        action: ControlAction | None = None,
    ) -> QualityAssessment:
        caveats: list[str] = []
        median = float(self.params["sulfur_median_train"])
        alpha = float(self.params["alpha"])
        reading = state.quality.get(self.METRIC)
        lab_value: float | None = None
        if reading is not None and reading.value is not None:
            lab_value = float(reading.value)
            age_h: float | None = (reading.age_minutes or 0.0) / 60.0
            caveats.append(f"Последний результат ЛИМС {lab_value:.1f} мг/кг, {age_h:.0f} ч назад.")
        else:
            age_h = None
            caveats.append("Нет лабораторного результата: прогноз — медиана обучающего периода.")

        ewma = state.data_flags.get("lab_sulfur_ewma")
        if ewma is not None:
            base = float(ewma)
        elif lab_value is not None:
            # An EWMA started at the training median that has seen one result.
            base = median + alpha * (lab_value - median)
            caveats.append("В состоянии нет истории анализов: сглаживание по одному результату.")
        else:
            base = median

        deltas = self._deltas(state, action)
        shift = self._what_if(state, deltas)
        mean = max(base + shift, 0.0)

        ref_gap = float(self.params["ref_gap_hours"])
        effective_age = age_h if age_h is not None else 4.0 * ref_gap
        widen = math.sqrt(max(effective_age, ref_gap) / ref_gap)
        p05 = max(mean + float(self.params["resid_q05"]) * widen, 0.0)
        p95 = mean + float(self.params["resid_q95"]) * widen

        confidence = self.BASE_CONFIDENCE
        if age_h is None:
            confidence *= 0.25
        else:
            confidence *= 1.0 / (1.0 + age_h / self.AGE_HALFLIFE_HOURS)
        pak = state.data_flags.get("pak_sulfur") or {}
        if pak.get("healthy") is False:
            confidence *= 0.5
            caveats.append("Поточный анализатор серы неисправен или заморожен.")

        return QualityAssessment(
            metrics={self.METRIC: round(mean, 4)},
            risk_of_spec_breach=_prob_above(p05, p95, self.SPEC_LIMIT_MG_KG),
            confidence=round(confidence, 4),
            horizon_minutes=int(round(ref_gap * 60)),
            features_used=[f"lims:{self.METRIC}", *sorted(deltas)],
            assumptions=[
                "Базовая модель: экспоненциальное сглаживание результатов ЛИМС, доступных на момент решения (задержка 4 ч).",
                "Интервал p05–p95 — квантили ошибки этого прогноза на обучающем периоде; расширяется с возрастом последнего анализа.",
                "Влияние рычагов: знаки по физике процесса, величины — допущения.",
                "Сера оценивается в гидроочищенном ДТ, а не в товарном.",
            ],
            details={
                "model": "ewma_v0",
                "intervals": {
                    self.METRIC: {"mean": round(mean, 4), "p05": round(p05, 4), "p95": round(p95, 4)}
                },
                "lab_value": lab_value,
                "ewma": round(base, 4),
                "alpha": alpha,
                "lab_age_hours": None if age_h is None else round(age_h, 2),
                "what_if_shift": round(shift, 4),
                "deltas": deltas,
                "caveats": caveats,
            },
        )

    @staticmethod
    def _deltas(state: ProcessState, action: ControlAction | None) -> dict[str, float]:
        if action is None or not action.changes:
            return {}
        deltas: dict[str, float] = {}
        for tag, new in action.changes.items():
            current = state.controllable.get(tag)
            if current is None or new is None:
                continue
            delta = float(new) - float(current)
            if delta != 0.0:
                deltas[tag] = round(delta, 6)
        return deltas

    def _what_if(self, state: ProcessState, deltas: dict[str, float]) -> float:
        if not deltas:
            return 0.0
        shift = sum(self.sensitivity.get(tag, 0.0) * delta for tag, delta in deltas.items())
        f2 = state.controllable.get("242000:F2")
        f26 = state.controllable.get("242000:F26")
        if f2 and f26:
            new_f2 = f2 + deltas.get("242000:F2", 0.0)
            new_f26 = f26 + deltas.get("242000:F26", 0.0)
            if new_f26 > 0:
                shift += self.ratio_sensitivity * (new_f2 / new_f26 - f2 / f26)
        return shift


def _prob_above(p05: float, p95: float, limit: float) -> float:
    """Crude P(value > limit): p05..p95 is a 90 % band, linear inside, thin tails outside."""
    span = p95 - p05
    if span <= 0:
        return 1.0 if p05 > limit else 0.0
    if limit >= p95:
        return round(max(0.0, 0.05 * (1.0 - (limit - p95) / span)), 4)
    if limit <= p05:
        return round(min(1.0, 0.95 + 0.05 * (p05 - limit) / span), 4)
    return round(0.05 + 0.90 * (p95 - limit) / span, 4)
