from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from neftecode.orchestration.orchestrator import Orchestrator

app = FastAPI(title="Neftecode MAS", version="0.1.0")
_orchestrator = Orchestrator()


class RunRequest(BaseModel):
    timestamp: datetime = Field(..., description="Decision timestamp")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run_cycle")
def run_cycle(body: RunRequest) -> dict:
    try:
        rec = _orchestrator.run_cycle(body.timestamp)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return rec.model_dump(mode="json")
