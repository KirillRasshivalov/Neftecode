from __future__ import annotations

import math

from neftecode.domain.actions import ControlAction
from neftecode.domain.agent_results import MetricInterval, QualityAssessment
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


class BreachClassifier:
    """P(lab sulfur above spec) for the regime as measured, from fitted coefficients.

    A **nowcast, not a forecast**. `scripts/analysis/breach_classifier.py` finds the
    signal only at horizon 0 - held-out AUC 0.72, against 0.48-0.53 at 4 to 24 hours -
    and it is carried by the online analyzer, not by the levers. Scoring is one dot
    product over standardised features, so the decision path needs no scikit-learn and
    the same state gives the same number to the last digit on every run.

    The agent reads no files: the caller loads
    `models/breach_classifier.json` and passes its `linear_model` block here.
    """

    #: How many contributions the operator card may name.
    TOP_TERMS = 3

    def __init__(self, model: dict) -> None:
        missing = [k for k in ("features", "mean", "scale", "coef", "intercept") if k not in model]
        if missing:
            raise ValueError(f"breach classifier is missing {missing}")
        self.features = [str(name) for name in model["features"]]
        self.mean = [float(v) for v in model["mean"]]
        self.scale = [float(v) or 1.0 for v in model["scale"]]
        self.coef = [float(v) for v in model["coef"]]
        self.intercept = float(model["intercept"])
        sizes = {len(self.features), len(self.mean), len(self.scale), len(self.coef)}
        if len(sizes) != 1:
            raise ValueError(f"breach classifier arrays disagree in length: {sorted(sizes)}")
        self.auc_heldout = model.get("auc_heldout")
        self.trigger_threshold = model.get("trigger_threshold")

    def evaluate(self, features: dict) -> tuple[float, list[dict]] | None:
        """Probability and its largest terms, or None when any feature is missing."""
        total = self.intercept
        terms: list[dict] = []
        for name, mean, scale, coef in zip(self.features, self.mean, self.scale, self.coef):
            raw = features.get(name)
            if raw is None:
                return None
            value = float(raw)
            if not math.isfinite(value):
                return None
            contribution = coef * (value - mean) / scale
            total += contribution
            terms.append({
                "feature": name,
                "value": round(value, 4),
                "contribution": round(contribution, 4),
            })
        probability = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, total))))
        terms.sort(key=lambda term: abs(term["contribution"]), reverse=True)
        return round(probability, 4), terms[: self.TOP_TERMS]


class QualityAgentBaseline:
    """EWMA baseline: smoothed lab history, empirical interval, signed what-if.

    Works whether or not the telemetry predicts sulfur. Consecutive lab results are
    only weakly autocorrelated, so the prediction is an exponentially weighted mean of
    past results (smoothing factor chosen on the training period), not the last
    result. The interval is the training residual spread of that predictor, so the
    p95 is real even without a trained model. The as-of EWMA is computed by the state
    builder and arrives in `data_flags["lab_sulfur_ewma"]`; parameters are fitted by
    `scripts/analysis/quality_baseline_fit.py` and passed in. The agent reads no files:
    every agent here is a pure function of the state and the parameters given to it.

    The what-if layer moves the prediction when a lever changes. It is multiplicative:
    product sulfur = level × exp(Σ sensitivity × Δlever / reference level). At the
    reference level (the training median) the local slope equals the stated sensitivity —
    a finite step moves it slightly less, −0.67 rather than −0.70 mg/kg for 2 °C of T5.
    Elsewhere it moves in proportion to the level, and the prediction can never go
    negative however large the move. Temperature enters as an exponential, as in
    Arrhenius kinetics; for pressure and flow the exponential matches the usual power
    law over the few-percent steps the optimizer takes. Directions follow the physics
    and are fixed; magnitudes are assumptions.

    A scenario may state a feed sulfur other than the one measured
    (`data_flags["feed_sulfur_pct_scenario"]`). The smoothed level was observed under
    the measured feed, so it is rescaled by (scenario / measured) ** exponent — first-
    order desulfurisation kinetics, product sulfur proportional to feed sulfur. This is
    what lets the organisers' example run: sulfur in the crude rises, and the unit has
    to compensate.
    """

    METRIC = "sulfur_mg_kg"
    SPEC_LIMIT_MG_KG = 10.0

    #: The organisers set the prediction horizon at 0 to 3 hours and the dead time
    #: between a control move and a quality change at the same 0 to 3 hours, so the
    #: assessment is reported at 3 hours. Two horizons live inside it and the trace
    #: keeps them apart: the risk is a 0-2 h nowcast, while the interval is still the
    #: 24 h lab-to-lab residual spread. Using a 24 h spread over 3 hours overstates the
    #: uncertainty, which widens p95 and so errs towards refusing — the safe direction.
    HORIZON_MINUTES = 180

    #: First-order kinetics: product sulfur proportional to feed sulfur.
    FEED_SULFUR_EXPONENT = 1.0

    #: mg/kg of sulfur per unit change of a lever **at the reference level**, the training
    #: median. Signs are physics, sizes are assumed.
    #: The pressure figure is the local slope of the usual power law, sulfur ~ P**-0.8,
    #: at 3.92 MPa and 8.5 mg/kg. It makes pressure a deliberately weak lever: the whole
    #: historical span of `P13` is 0.42 MPa, so one step cannot do what a step of `T5` does.
    DEFAULT_SENSITIVITY: dict[str, float] = {
        "242000:T5": -0.35,     # hotter reactor: deeper desulfurisation
        "242000:F26": 0.05,     # more throughput over the same catalyst: less contact time
        "242000:P13": -1.75,    # higher hydrogen partial pressure: deeper desulfurisation
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
        classifier: "BreachClassifier | None" = None,
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
        # Optional by design: with no classifier the agent behaves exactly as before.
        self.classifier = classifier

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

        # Scenario: a feed sulfur other than the measured one. The smoothed level was
        # observed under the measured feed, so it scales with the ratio.
        feed_now = state.data_flags.get("feed_sulfur_pct")
        feed_scenario = state.data_flags.get("feed_sulfur_pct_scenario")
        feed_factor = 1.0
        if feed_scenario is not None and feed_now:
            feed_factor = (float(feed_scenario) / float(feed_now)) ** self.FEED_SULFUR_EXPONENT
            caveats.append(
                f"Сценарий: сера в сырье {float(feed_scenario):.2f} % вместо измеренных "
                f"{float(feed_now):.2f} % — уровень серы пересчитан ×{feed_factor:.2f}."
            )
        level = base * feed_factor

        deltas = self._deltas(state, action)
        mean = level * math.exp(self._what_if(state, deltas) / median)
        shift = mean - level

        ref_gap = float(self.params["ref_gap_hours"])
        effective_age = age_h if age_h is not None else 4.0 * ref_gap
        widen = math.sqrt(max(effective_age, ref_gap) / ref_gap)
        q05 = float(self.params["resid_q05"]) * widen
        q95 = float(self.params["resid_q95"]) * widen
        p05 = max(mean + q05, 0.0)
        p95 = mean + q95

        # The interval keeps the level and the hard p95 check. The classifier, when it
        # is allowed to speak, keeps the risk - but it cannot answer a what-if, because
        # the levers are noise to it. So the effect of an action stays with the physics
        # layer and enters as an increment over the classifier reading of the regime.
        hold_mean = level
        hold_risk = _prob_above(max(hold_mean + q05, 0.0), hold_mean + q95, self.SPEC_LIMIT_MG_KG)
        interval_risk = _prob_above(p05, p95, self.SPEC_LIMIT_MG_KG)
        probability, terms, unavailable = self._classify(state)
        if probability is None:
            risk, risk_source = interval_risk, "interval"
            if self.classifier is not None:
                caveats.append(f"Вероятность нарушения — оценка по интервалу: {unavailable}.")
        else:
            risk = round(min(1.0, max(0.0, probability + interval_risk - hold_risk)), 4)
            risk_source = "classifier"
            caveats.append(
                f"Вероятность нарушения {probability:.0%} — классификатор по поточному "
                "анализатору и телеметрии; это оценка на сейчас, не прогноз."
            )

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
            risk_of_spec_breach=risk,
            confidence=round(confidence, 4),
            horizon_minutes=self.HORIZON_MINUTES,
            features_used=[
                f"lims:{self.METRIC}",
                *sorted(deltas),
                *(["model:breach_classifier"] if risk_source == "classifier" else []),
            ],
            assumptions=[
                "Базовая модель: экспоненциальное сглаживание результатов ЛИМС, доступных на момент решения (задержка 4 ч).",
                "Интервал p05–p95 — квантили ошибки этого прогноза на обучающем периоде; расширяется с возрастом последнего анализа.",
                "Влияние рычагов мультипликативное: знаки по физике процесса, величины — допущения.",
                "Сера оценивается в гидроочищенном ДТ, а не в товарном.",
                "Вероятность нарушения: классификатор при исправном анализаторе, иначе оценка "
                "по интервалу. Эффект действия в обоих случаях даёт физический слой, не модель.",
            ],
            intervals={
                self.METRIC: MetricInterval(mean=round(mean, 4), p05=round(p05, 4), p95=round(p95, 4)),
            },
            details={
                "model": "ewma_v0",
                "intervals": {
                    self.METRIC: {"mean": round(mean, 4), "p05": round(p05, 4), "p95": round(p95, 4)}
                },
                "lab_value": lab_value,
                "ewma": round(base, 4),
                "alpha": alpha,
                "lab_age_hours": None if age_h is None else round(age_h, 2),
                "level": round(level, 4),
                "feed_factor": round(feed_factor, 4),
                "feed_sulfur_pct": None if feed_now is None else round(float(feed_now), 4),
                "feed_sulfur_pct_scenario": None if feed_scenario is None else round(float(feed_scenario), 4),
                "what_if_shift": round(shift, 4),
                "deltas": deltas,
                "risk_source": risk_source,
                "risk_interval": interval_risk,
                "risk_interval_hold": hold_risk,
                "risk_classifier_hold": probability,
                "risk_horizon_minutes": 120 if risk_source == "classifier" else int(round(ref_gap * 60)),
                "interval_horizon_minutes": int(round(ref_gap * 60)),
                "classifier_terms": terms,
                "classifier_unavailable": unavailable,
                "classifier_auc_heldout": self.classifier.auc_heldout if self.classifier else None,
                "caveats": caveats,
            },
        )

    def _classify(self, state: ProcessState) -> tuple[float | None, list[dict], str | None]:
        """Probability, its largest terms, and - when there is none - the reason why.

        The analyzer carries the whole signal, so a frozen or stale one disqualifies the
        model rather than degrading it quietly. Same for a stopped unit and for missing
        feature windows.
        """
        if self.classifier is None:
            return None, [], "классификатор не подключён"
        if state.data_flags.get("feed_sulfur_pct_scenario") is not None:
            # The classifier reads the analyzer that is really there; a hypothetical feed
            # has no measurement behind it, so the risk comes from the shifted interval.
            return None, [], "сценарий с изменённой серой в сырье: анализатор его не видит"
        if state.data_flags.get("running") is not True:
            return None, [], "установка не работает"
        if (state.data_flags.get("pak_sulfur") or {}).get("healthy") is not True:
            return None, [], "поточный анализатор неисправен или заморожен"
        windows = state.windows_for_agents()
        if not isinstance(windows, dict):
            return None, [], "в состоянии нет окон телеметрии"
        scored = self.classifier.evaluate(windows)
        if scored is None:
            return None, [], "часть признаков недоступна"
        probability, terms = scored
        return probability, terms, None

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
        """Sum of sensitivity × Δlever, in mg/kg at the reference level.

        `assess` turns it into the multiplicative factor exp(sum / reference level).
        """
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
