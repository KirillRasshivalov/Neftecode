from __future__ import annotations

from pydantic import BaseModel, Field


class ControlAction(BaseModel):
    changes: dict[str, float] = Field(default_factory=dict)
    label: str | None = None

    def is_noop(self) -> bool:
        return not self.changes
