from __future__ import annotations

from datetime import datetime

from neftecode.data.config import load_tags_whitelist
from neftecode.domain.state import ProcessState, QualityReading, TagValue


class ProcessStateBuilder:
    def __init__(self, whitelist: dict | None = None) -> None:
        self.whitelist = whitelist or load_tags_whitelist()

    def build(self, timestamp: datetime, scenario: str | None = None) -> ProcessState:
        controllable: dict[str, float] = {}
        kip: dict[str, TagValue] = {}

        for item in self.whitelist.get("controllable_parameters", []):
            tag = item["tag"]
            lo = float(item["range"]["min"])
            hi = float(item["range"]["max"])
            value = (lo + hi) / 2.0
            controllable[tag] = value
            kip[tag] = TagValue(tag=tag, value=value, unit=item.get("unit"), source=item.get("current_source"))

        sulfur_value = 8.0
        age_minutes = 30.0
        complete = True
        notes = ["ProcessStateBuilder is using scaffold defaults until raw data is wired."]

        if scenario == "risk_sulfur":
            sulfur_value = 11.5
            notes.append("Demo scenario risk_sulfur: elevated sulfur.")
        elif scenario == "stale_data":
            age_minutes = 10_000.0
            notes.append("Demo scenario stale_data: intentionally stale LIMS.")
        elif scenario == "normal":
            notes.append("Demo scenario normal: stable regime.")

        quality = {
            "sulfur_mg_kg": QualityReading(
                metric="sulfur_mg_kg",
                value=sulfur_value,
                unit="mg/kg",
                source="lims",
                measured_at=timestamp,
                age_minutes=age_minutes,
            )
        }

        return ProcessState(
            timestamp=timestamp,
            kip=kip,
            quality=quality,
            controllable=controllable,
            data_flags={"scaffold_mode": True, "complete": complete, "scenario": scenario},
            notes=notes,
        )
