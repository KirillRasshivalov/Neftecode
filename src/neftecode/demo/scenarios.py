from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class DemoScenario:
    name: str
    timestamp: datetime
    description: str


DEMO_SCENARIOS: dict[str, DemoScenario] = {
    "normal": DemoScenario(
        name="normal",
        timestamp=datetime(2024, 6, 1, 12, 0, 0),
        description="Устойчивый период: система не должна навязывать лишние действия.",
    ),
    "risk_sulfur": DemoScenario(
        name="risk_sulfur",
        timestamp=datetime(2024, 8, 15, 8, 0, 0),
        description="Период с риском ухудшения качества / смены режима.",
    ),
    "stale_data": DemoScenario(
        name="stale_data",
        timestamp=datetime(2025, 1, 10, 18, 0, 0),
        description="Неполные, устаревшие или аномальные данные → ожидается отказ.",
    ),
    "full_cycle": DemoScenario(
        name="full_cycle",
        timestamp=datetime(2024, 11, 20, 14, 30, 0),
        description="Полный цикл взаимодействия агентов до финальной рекомендации.",
    ),
}


def get_scenario(name: str) -> DemoScenario:
    try:
        return DEMO_SCENARIOS[name]
    except KeyError as exc:
        known = ", ".join(DEMO_SCENARIOS)
        raise KeyError(f"Unknown scenario '{name}'. Known: {known}") from exc
