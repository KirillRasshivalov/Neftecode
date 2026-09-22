"""Blending agent: the cheapest commercial blend that meets the product specification.

The organisers' scheme «Блендинг из резервуаров»: three components with shares and
stocks, a cetane improver with a dose, shares summing to 100 %, and a commercial tank
held to sulfur ≤ 10 mg/kg, T95 ≤ 360 °C, cetane ≥ 51 (49 in winter) and density
820-845 kg/m³ (800-845 in winter).

Same pattern as the lever optimizer: generate every blend on a grid, drop the ones that
break a limit, rank what is left. A limit is never traded against cost: the hard
constraints are a filter, not a term in the score. With no feasible blend the agent refuses and names the limit that cannot be
met, and by how much.

Mixing, by mass share w:

- sulfur is mass ppm, so it mixes linearly by mass — exact;
- density mixes by volume, 1/ρ = Σ w/ρ — exact for ideal mixing;
- T95 and cetane mix linearly by **volume** share — an assumption: refineries use
  blending indices for both. The organisers allow a simple, linear model.

The tank averages many hours of production, so the hydrotreated component enters
with the regime's **predicted mean** sulfur, not the p95 of a single lab sample. The
p95 stays where it belongs, on the hard check of the hydrotreating regime.

The agent is a pure function: components, specification and
additives arrive through the constructor, the diesel sulfur of the regime as an
argument. It reads no files.
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field

PRIMARY = "hydrotreated_diesel"

#: A limit counts as binding when the optimum sits this close to it. Natural units.
BINDING_TOLERANCE = {"sulfur_mg_kg": 0.2, "density_kg_m3": 1.0, "t95_c": 2.0, "cetane": 0.3}

LABELS = {
    "sulfur_mg_kg": ("сера", "мг/кг"),
    "density_kg_m3": ("плотность", "кг/м³"),
    "t95_c": ("T95", "°C"),
    "cetane": ("цетановое число", ""),
}


class BlendComponent(BaseModel):
    key: str
    label: str
    sulfur_mg_kg: float
    density_kg_m3: float
    t95_c: float
    cetane: float
    stock_t: float
    price_rel: float = 1.0
    sources: dict[str, str] = Field(default_factory=dict)


class Additive(BaseModel):
    """Cetane improver with a saturating response: gain = G · (1 − exp(−dose / d0))."""

    key: str
    label: str
    price_rel_per_t: float
    max_dose_kg_t: float
    cetane_gain_max: float
    dose_scale_kg_t: float

    def gain(self, dose_kg_t: float) -> float:
        if dose_kg_t <= 0.0 or self.cetane_gain_max <= 0.0:
            return 0.0
        return self.cetane_gain_max * (1.0 - math.exp(-dose_kg_t / self.dose_scale_kg_t))

    def dose_for(self, needed: float) -> float | None:
        """Smallest dose that adds `needed` cetane numbers, or None if none can."""
        if needed <= 0.0:
            return 0.0
        if needed >= self.cetane_gain_max:
            return None
        dose = -self.dose_scale_kg_t * math.log(1.0 - needed / self.cetane_gain_max)
        return dose if dose <= self.max_dose_kg_t else None

    def cost_rel_per_t(self, dose_kg_t: float) -> float:
        return dose_kg_t / 1000.0 * self.price_rel_per_t


class BlendSpec(BaseModel):
    season: Literal["summer", "winter"]
    sulfur_mg_kg_max: float
    t95_c_max: float
    cetane_min: float
    density_min: float
    density_max: float


class BlendCandidate(BaseModel):
    shares: dict[str, float]
    tonnes: dict[str, float]
    additive: str | None
    additive_kg_t: float
    properties: dict[str, float]
    margins: dict[str, float]
    cost_rel_per_t: float
    feasible: bool
    violations: list[str]


class BlendRecommendation(BaseModel):
    outcome: Literal["blend", "refuse"]
    diesel_sulfur_mg_kg: float
    best: BlendCandidate | None = None
    alternatives: list[BlendCandidate] = Field(default_factory=list)
    closest_infeasible: BlendCandidate | None = None
    binding: list[str] = Field(default_factory=list)
    refuse_reason: str | None = None
    n_candidates: int
    n_feasible: int
    spec: BlendSpec
    assumptions: list[str] = Field(default_factory=list)


class SulfurBudget(BaseModel):
    """What the blending agent tells the hydrotreating side before it decides.

    The largest sulfur in the hydrotreated diesel for which the tank can still be made
    to specification — without additive, with no more than `max_dilution_share` of
    diluent, and with `margin_mg_kg` of room under the tank's own limit. None when no
    such blend exists at all.
    """

    budget_mg_kg: float | None
    shares: dict[str, float] | None = None
    margin_mg_kg: float
    max_dilution_share: float
    reason: str | None = None


def build_components(derived: dict, config: dict) -> list[BlendComponent]:
    """Merge properties derived from the lab (`models/blend_components.json`) with
    what only the configuration knows: stocks, prices and the assumed sulfur."""
    out = []
    for key, settings in config["components"].items():
        found = derived["components"][key]
        props = dict(found["properties"])
        sources = dict(found.get("sources", {}))
        if "sulfur_mg_kg" in settings:
            props["sulfur_mg_kg"] = float(settings["sulfur_mg_kg"])
            sources["sulfur_mg_kg"] = "допущение, configs/blending.yaml"
        out.append(BlendComponent(
            key=key,
            label=found["label"],
            sulfur_mg_kg=props["sulfur_mg_kg"],
            density_kg_m3=props["density_kg_m3"],
            t95_c=props["t95_c"],
            cetane=props["cetane"],
            stock_t=float(settings["stock_t"]),
            price_rel=float(settings.get("price_rel", 1.0)),
            sources=sources,
        ))
    return out


def build_spec(config: dict, season: str | None = None) -> BlendSpec:
    season = season or config.get("season", "summer")
    spec = config["spec"]
    lo, hi = spec["density_kg_m3"][season]
    return BlendSpec(
        season=season,
        sulfur_mg_kg_max=float(spec["sulfur_mg_kg_max"]),
        t95_c_max=float(spec["t95_c_max"]),
        cetane_min=float(spec["cetane_min"][season]),
        density_min=float(lo),
        density_max=float(hi),
    )


def build_additives(config: dict) -> list[Additive]:
    return [
        Additive(key=key, **{k: v for k, v in settings.items() if k != "enabled"})
        for key, settings in (config.get("additives") or {}).items()
        if settings.get("enabled", True)
    ]


class BlendingAgent:
    ASSUMPTIONS = [
        "Смешение: сера — линейно по массе, плотность — по объёму (точно); T95 и цетановое "
        "число — линейно по объёму (допущение: на НПЗ смешивают по индексам).",
        "Гидроочищенный компонент идёт в смесь со средним прогнозом серы режима: резервуар "
        "усредняет многие часы производства.",
        "Цены компонентов равны: ценовых данных в пакете нет. Экономику задаёт присадка — "
        "в 100 раз дороже тонны ДТ.",
        "Отклик присадки насыщается: +G·(1 − exp(−доза/d0)). Величины G и d0 — допущения.",
    ]

    def __init__(
        self,
        components: list[BlendComponent],
        spec: BlendSpec,
        additives: list[Additive],
        batch_t: float,
        grid_step_pct: int = 5,
        budget_margin_mg_kg: float = 0.5,
        max_dilution_share: float = 0.3,
    ) -> None:
        if not components:
            raise ValueError("blending needs at least one component")
        if 100 % grid_step_pct:
            raise ValueError("grid_step_pct must divide 100")
        self.components = components
        self.spec = spec
        self.additives = additives
        self.batch_t = float(batch_t)
        self.units = 100 // grid_step_pct
        self.budget_margin_mg_kg = float(budget_margin_mg_kg)
        self.max_dilution_share = float(max_dilution_share)

    # ---------------------------------------------------------------- public

    def labels(self) -> list[str]:
        s = self.spec
        season = "лето" if s.season == "summer" else "зима"
        return [
            f"товарная смесь: сера ≤ {s.sulfur_mg_kg_max:g} мг/кг",
            f"товарная смесь: T95 ≤ {s.t95_c_max:g} °C",
            f"товарная смесь: цетановое число ≥ {s.cetane_min:g} ({season})",
            f"товарная смесь: плотность {s.density_min:g}–{s.density_max:g} кг/м³ ({season})",
            "товарная смесь: доли в сумме 100 %, в пределах запасов",
        ]

    def blend(self, diesel_sulfur_mg_kg: float | None = None) -> BlendRecommendation:
        components = self._with_diesel_sulfur(diesel_sulfur_mg_kg)
        diesel = next((c for c in components if c.key == PRIMARY), components[0])
        grid = list(self._within_stock(components))
        base = dict(
            diesel_sulfur_mg_kg=round(diesel.sulfur_mg_kg, 4),
            spec=self.spec,
            assumptions=list(self.ASSUMPTIONS),
        )
        if not grid:
            total = sum(c.stock_t for c in components)
            return BlendRecommendation(
                outcome="refuse",
                refuse_reason=(
                    f"Запасов не хватает на партию: всего {total:g} т при партии {self.batch_t:g} т."
                ),
                n_candidates=0,
                n_feasible=0,
                **base,
            )

        candidates = [self._evaluate(shares, components) for shares in grid]
        feasible = sorted((c for c in candidates if c.feasible), key=self._rank)
        if not feasible:
            closest = min(candidates, key=self._shortfall)
            return BlendRecommendation(
                outcome="refuse",
                closest_infeasible=closest,
                refuse_reason=(
                    "Допустимой смеси нет: ни одна комбинация долей и дозы присадки не проходит "
                    "спецификацию. Ближайшая: " + "; ".join(closest.violations) + "."
                ),
                n_candidates=len(candidates),
                n_feasible=0,
                **base,
            )

        best = feasible[0]
        return BlendRecommendation(
            outcome="blend",
            best=best,
            alternatives=self._alternatives(best, feasible),
            binding=[
                LABELS[name][0]
                for name, margin in best.margins.items()
                if margin <= BINDING_TOLERANCE[name]
            ],
            n_candidates=len(candidates),
            n_feasible=len(feasible),
            **base,
        )

    def sulfur_budget(self) -> SulfurBudget:
        """How much sulfur the hydrotreated diesel may carry and the tank still meet spec.

        Every blend on the grid that the stocks allow, that keeps at least
        1 − `max_dilution_share` of the unit's own product and that passes density,
        T95 and cetane **without** additive, tolerates diesel sulfur up to
        (tank limit − margin − sulfur brought by the diluents) / diesel share. The
        budget is the largest of those.

        Why the two limits. Without the dilution cap the budget comes out of blends
        that are mostly diluent — the arithmetic allows a hydrotreater at 16 mg/kg if
        only 30 % of the tank is its product, but the unit makes diesel continuously and
        cannot park the rest. Without the no-additive rule the budget would be bought
        with an additive at a hundred times the price of the diesel.
        """
        by_key = {c.key: c for c in self.components}
        ceiling = self.spec.sulfur_mg_kg_max - self.budget_margin_mg_kg
        best: float | None = None
        best_shares: dict[str, float] | None = None
        for shares in self._within_stock(self.components):
            own = shares.get(PRIMARY, 0.0)
            if own <= 0.0 or own < 1.0 - self.max_dilution_share - 1e-9:
                continue
            props = self._mix(shares, by_key)
            if not self._meets_all_but_sulfur(props):
                continue
            diluent = sum(w * by_key[k].sulfur_mg_kg for k, w in shares.items() if k != PRIMARY)
            allowed = (ceiling - diluent) / own
            if best is None or allowed > best + 1e-12:
                best, best_shares = allowed, shares
        common = dict(margin_mg_kg=self.budget_margin_mg_kg, max_dilution_share=self.max_dilution_share)
        if best is None:
            return SulfurBudget(
                budget_mg_kg=None,
                reason="ни одна смесь без присадки и с разрешённой долей разбавителя не проходит спецификацию",
                **common,
            )
        return SulfurBudget(
            budget_mg_kg=round(best, 3),
            shares={k: round(w, 4) for k, w in best_shares.items()},
            **common,
        )

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _mix(shares: dict[str, float], by_key: dict[str, BlendComponent]) -> dict[str, float]:
        """Blend properties with no additive: sulfur by mass, the rest by volume."""
        volume = sum(w / by_key[k].density_kg_m3 for k, w in shares.items())
        vol_share = {k: (w / by_key[k].density_kg_m3) / volume for k, w in shares.items()}
        return {
            "sulfur_mg_kg": sum(w * by_key[k].sulfur_mg_kg for k, w in shares.items()),
            "density_kg_m3": 1.0 / volume,
            "t95_c": sum(v * by_key[k].t95_c for k, v in vol_share.items()),
            "cetane": sum(v * by_key[k].cetane for k, v in vol_share.items()),
        }

    def _meets_all_but_sulfur(self, props: dict[str, float]) -> bool:
        s = self.spec
        return (
            s.density_min <= props["density_kg_m3"] <= s.density_max
            and props["t95_c"] <= s.t95_c_max
            and props["cetane"] >= s.cetane_min
        )

    def _with_diesel_sulfur(self, sulfur: float | None) -> list[BlendComponent]:
        if sulfur is None:
            return self.components
        return [
            c.model_copy(update={"sulfur_mg_kg": float(sulfur)}) if c.key == PRIMARY else c
            for c in self.components
        ]

    def _within_stock(self, components: list[BlendComponent]):
        """Every share vector on the grid that the tanks can actually supply."""
        for units in _compositions(self.units, len(components)):
            shares = [u / self.units for u in units]
            if all(w * self.batch_t <= c.stock_t + 1e-9 for w, c in zip(shares, components)):
                yield dict(zip((c.key for c in components), shares))

    def _evaluate(self, shares: dict[str, float], components: list[BlendComponent]) -> BlendCandidate:
        by_key = {c.key: c for c in components}
        mixed = self._mix(shares, by_key)
        sulfur, density = mixed["sulfur_mg_kg"], mixed["density_kg_m3"]
        t95, cetane = mixed["t95_c"], mixed["cetane"]

        additive, dose = self._additive_for(self.spec.cetane_min - cetane)
        if additive is not None:
            cetane += additive.gain(dose)

        props = {"sulfur_mg_kg": sulfur, "density_kg_m3": density, "t95_c": t95, "cetane": cetane}
        margins = {
            "sulfur_mg_kg": self.spec.sulfur_mg_kg_max - sulfur,
            "density_kg_m3": min(density - self.spec.density_min, self.spec.density_max - density),
            "t95_c": self.spec.t95_c_max - t95,
            "cetane": cetane - self.spec.cetane_min,
        }
        violations = [self._violation(name, props[name], margin) for name, margin in margins.items() if margin < -1e-9]
        cost = sum(w * by_key[k].price_rel for k, w in shares.items())
        if additive is not None:
            cost += additive.cost_rel_per_t(dose)
        return BlendCandidate(
            shares={k: round(w, 4) for k, w in shares.items()},
            tonnes={k: round(w * self.batch_t, 2) for k, w in shares.items()},
            additive=None if additive is None or dose == 0.0 else additive.key,
            additive_kg_t=round(dose if additive is not None else 0.0, 3),
            properties={k: round(v, 3) for k, v in props.items()},
            margins={k: round(v, 3) for k, v in margins.items()},
            cost_rel_per_t=round(cost, 5),
            feasible=not violations,
            violations=violations,
        )

    def _additive_for(self, shortfall: float) -> tuple[Additive | None, float]:
        """The cheapest enabled additive that closes the cetane shortfall, and its dose.

        When none can close it, the strongest dose of the first additive is applied so
        the violation that remains is the smallest one achievable.
        """
        if shortfall <= 0.0 or not self.additives:
            return None, 0.0
        options = []
        for additive in self.additives:
            dose = additive.dose_for(shortfall)
            if dose is not None:
                options.append((additive.cost_rel_per_t(dose), additive.key, additive, dose))
        if options:
            _, _, additive, dose = min(options)
            return additive, dose
        strongest = max(self.additives, key=lambda a: a.gain(a.max_dose_kg_t))
        return strongest, strongest.max_dose_kg_t

    def _violation(self, name: str, value: float, margin: float) -> str:
        word, unit = LABELS[name]
        s = self.spec
        limit = {
            "sulfur_mg_kg": f"> {s.sulfur_mg_kg_max:g}",
            "t95_c": f"> {s.t95_c_max:g}",
            "cetane": f"< {s.cetane_min:g}",
            "density_kg_m3": f"вне {s.density_min:g}–{s.density_max:g}",
        }[name]
        return f"{word} {value:.1f} {limit}{(' ' + unit) if unit else ''} (на {abs(margin):.1f})"

    @staticmethod
    def _rank(c: BlendCandidate) -> tuple:
        # 1. cheaper; 2. more of the unit's own product — change as little as possible;
        # 3. more room to the nearest limit, measured in binding tolerances.
        room = min(c.margins[k] / BINDING_TOLERANCE[k] for k in BINDING_TOLERANCE)
        return (c.cost_rel_per_t, -c.shares.get(PRIMARY, 0.0), -room)

    @staticmethod
    def _shortfall(c: BlendCandidate) -> float:
        return sum(-min(0.0, c.margins[k]) / BINDING_TOLERANCE[k] for k in BINDING_TOLERANCE)

    @staticmethod
    def _alternatives(best: BlendCandidate, feasible: list[BlendCandidate]) -> list[BlendCandidate]:
        """Up to two runners-up that differ from the best in which components they use."""
        used = {k for k, w in best.shares.items() if w > 0}
        out, seen = [], {frozenset(used)}
        for c in feasible[1:]:
            mix = frozenset(k for k, w in c.shares.items() if w > 0)
            if mix not in seen:
                out.append(c)
                seen.add(mix)
            if len(out) == 2:
                break
        return out


def _compositions(total: int, parts: int):
    """Every way to write `total` as an ordered sum of `parts` non-negative integers."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total, -1, -1):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)
