"""Throughput and energy proxies for the multi-criteria comparison.

The task statement asks the optimizer to weigh, besides quality, **output** and
**energy or a transparent cost proxy** (ТЗ §4). The package carries no economic data, so
there is no currency here: output is the measured product flow, and energy is a
dimensionless index equal to 1.0 at the median regime of the training period.

Both are reported as a **percentage change against the current regime**, which is the
scale the ranking weights in `configs/constraints.yaml` are calibrated on. There the
weights are deliberately small: economics separates candidates that are close on quality
and equipment risk, and one lever step can never move the score by as much as the hold
margin. The task statement's priority — quality over economics — therefore holds in the
ranking as well as in the hard filter.

The energy index has two terms, each built from levers whose physical quantity the
corrected 24-2000 reference states:

* **fired heater** — duty to bring the gas/feed mixture to reaction temperature. Per unit
  of feed that duty is proportional to the temperature rise, hence
  `(T5 - heat_reference_c)`.
* **recycle compressor** — work to circulate the gas. Per unit of *product* it is
  proportional to the gas-to-feed ratio times the discharge pressure, hence
  `(F2 / F26) * P13`. Over a lever step of 0.05 MPa on roughly 3.9 MPa, treating the
  compressor head as linear in pressure is accurate to far better than the weights.

Because the index is specific — per cubic metre of product — raising the feed lowers it
while raising output. Both readings are true and they are reported separately, so the
ranking never hides one behind the other. Two things stop the optimizer from simply
pushing feed up: the hard sulfur filter, and the orchestrator's rule that a quality
trigger may not be answered by a step that raises sulfur.

Parameters come from `models/economics_reference.json`, fitted by
`scripts/analysis/economics_reference.py`. This module reads no files and holds no state: like every agent here it is a pure
function of what it is handed.
"""
from __future__ import annotations

from dataclasses import dataclass

from neftecode.domain.actions import ControlAction
from neftecode.domain.state import ProcessState

#: Below this the energy index is too close to zero for a percentage to mean anything.
#: A reactor colder than `heat_reference_c` is rejected outright: the heater term turns
#: negative and the proxy stops describing energy at all. Callers get zeros and the card
#: gets no effect block, which is what a stopped or barely warm unit deserves.
MIN_INDEX = 0.2

#: Hours per day, for stating output in tonnes per day as well as m³/h.
HOURS_PER_DAY = 24.0

NEUTRAL_METRICS = {"throughput": 0.0, "energy_or_cost_proxy": 0.0}


@dataclass(frozen=True)
class ProcessEconomics:
    """Output and energy effect of a candidate action, in percent of the current regime."""

    temperature_tag: str
    feed_tag: str
    gas_tag: str
    pressure_tag: str
    t5_median_c: float
    gas_term_median: float
    heat_reference_c: float
    w_heat: float
    w_compression: float
    density_t_m3: float

    @classmethod
    def from_reference(cls, reference: dict) -> "ProcessEconomics":
        """Build from `models/economics_reference.json`."""
        return cls(
            temperature_tag=reference["temperature_tag"],
            feed_tag=reference["feed_tag"],
            gas_tag=reference["gas_tag"],
            pressure_tag=reference["pressure_tag"],
            t5_median_c=float(reference["t5_median_c"]),
            gas_term_median=float(reference["gas_term_median"]),
            heat_reference_c=float(reference["heat_reference_c"]),
            w_heat=float(reference["w_heat"]),
            w_compression=float(reference["w_compression"]),
            density_t_m3=float(reference["density_t_m3"]),
        )

    # ------------------------------------------------------------------ reading

    def _current(self, state: ProcessState, tag: str) -> float | None:
        value = state.controllable.get(tag)
        if value is None:
            reading = state.kip.get(tag)
            value = None if reading is None else reading.value
        return None if value is None else float(value)

    def _regime(
        self, state: ProcessState, action: ControlAction | None
    ) -> tuple[float, float, float, float] | None:
        """(T5, F2, F26, P13) with the action's changes applied, or None if a tag is missing."""
        values = []
        changes = action.changes if action is not None else {}
        for tag in (self.temperature_tag, self.gas_tag, self.feed_tag, self.pressure_tag):
            value = changes.get(tag, self._current(state, tag))
            if value is None:
                return None
            values.append(float(value))
        t5, f2, feed, p13 = values
        return None if feed <= 0.0 else (t5, f2, feed, p13)

    # ------------------------------------------------------------------ index

    def terms(self, t5: float, f2: float, feed: float, p13: float) -> tuple[float, float]:
        """The two weighted terms of the specific energy index."""
        heat = (t5 - self.heat_reference_c) / (self.t5_median_c - self.heat_reference_c)
        gas = (f2 / feed) * p13 / self.gas_term_median
        return self.w_heat * heat, self.w_compression * gas

    def index(self, t5: float, f2: float, feed: float, p13: float) -> float:
        heat, gas = self.terms(t5, f2, feed, p13)
        return heat + gas

    def _usable(self, regime: tuple[float, float, float, float] | None) -> float | None:
        """The index of a regime the proxy can describe, else None."""
        if regime is None or regime[0] <= self.heat_reference_c:
            return None
        base = self.index(*regime)
        return None if base < MIN_INDEX else base

    # ------------------------------------------------------------------ output

    def metrics(self, state: ProcessState, action: ControlAction | None) -> dict[str, float]:
        """Ranking metrics: percent change of output and of specific energy."""
        before = self._regime(state, None)
        after = self._regime(state, action)
        base = self._usable(before)
        if base is None or after is None:
            return dict(NEUTRAL_METRICS)
        return {
            "throughput": (after[2] / before[2] - 1.0) * 100.0,
            "energy_or_cost_proxy": (self.index(*after) / base - 1.0) * 100.0,
        }

    def effect(self, state: ProcessState, action: ControlAction | None) -> dict | None:
        """The card's expected effect on output and energy, or None when not computable."""
        before = self._regime(state, None)
        after = self._regime(state, action)
        base = self._usable(before)
        if base is None or after is None:
            return None
        feed_before, feed_after = before[2], after[2]
        heat_before, gas_before = self.terms(*before)
        heat_after, gas_after = self.terms(*after)
        return {
            "output": {
                "tag": self.feed_tag,
                "unit": "м³/ч",
                "from": round(feed_before, 2),
                "to": round(feed_after, 2),
                "pct": round((feed_after / feed_before - 1.0) * 100.0, 3),
                "delta_t_day": round(
                    (feed_after - feed_before) * self.density_t_m3 * HOURS_PER_DAY, 2
                ),
            },
            "energy": {
                "unit": "индекс, 1.0 = типичный режим обучающего периода",
                "from": round(base, 4),
                "to": round(self.index(*after), 4),
                "pct": round((self.index(*after) / base - 1.0) * 100.0, 3),
                "heater_delta": round(heat_after - heat_before, 4),
                "compressor_delta": round(gas_after - gas_before, 4),
            },
        }
