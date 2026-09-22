"""An export that lacks a tag: refuse and name it, never crash.

The task statement: when data is insufficient the system must refuse and explain why.
A tag missing from the export entirely is the bluntest case of that. Three layers:

- the loader, on a tiny synthetic cache: strict by default so training cannot silently
  write NaN into a model, tolerant on request for the decision path;
- the orchestrator, on a hand-built state: an absent lever is a refusal naming it, an
  absent context tag is not;
- end to end on the real cache, when it is present.
"""
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # `scripts` lives at the root

from scripts import realdata as rd  # noqa: E402
from neftecode.data.config import load_constraints  # noqa: E402
from neftecode.domain.state import ProcessState, QualityReading  # noqa: E402
from neftecode.orchestration.orchestrator import Orchestrator  # noqa: E402

T = datetime(2026, 7, 18, 16, 0)


# ------------------------------------------------------------------ the loader


@pytest.fixture
def tiny_cache(tmp_path, monkeypatch):
    """A telemetry cache with T5, F26 and F2 but no P13, one sentinel in it."""
    frame = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=4, freq="10min"),
        "T5": [370.0, 371.0, 307.0, 372.0],   # 307 is a sentinel
        "F26": [250.0, 251.0, 252.0, 253.0],
        "F2": [90_000.0, 90_100.0, 90_200.0, 90_300.0],
    })
    frame.to_parquet(tmp_path / "tags_242000.parquet")
    monkeypatch.setattr(rd, "CACHE_DIR", tmp_path)
    return tmp_path


def test_a_tag_missing_from_the_export_is_reported(tiny_cache):
    assert rd.absent_tags(("T5", "F26", "F2", "P13")) == ["242000:P13"]


def test_by_default_the_loader_stops_and_names_the_tag(tiny_cache):
    with pytest.raises(KeyError, match="242000:P13"):
        rd.load_telemetry()


def test_on_request_the_loader_returns_the_tag_empty(tiny_cache):
    frame = rd.load_telemetry(allow_absent=True)
    assert list(frame.columns) == ["242000:T5", "242000:F26", "242000:F2", "242000:P13"]
    assert frame["242000:P13"].isna().all()
    # Everything else is loaded as usual, sentinels included.
    assert np.isnan(frame["242000:T5"].iloc[2])
    assert frame["242000:F26"].iloc[0] == 250.0


# ------------------------------------------------------------------ the orchestrator

WHITELIST = {
    "controllable_parameters": [
        {"tag": "242000:T5", "range": {"min": 347.0, "max": 388.0, "assumption": True}, "deltas": [-2.0, 0.0, 2.0]},
    ]
}


class FixedBuilder:
    def __init__(self, state):
        self.state = state

    def build(self, timestamp, scenario=None):
        return self.state


def make_state(absent):
    flags = {"complete": True, "running": True}
    if absent:
        flags["absent_tags"] = absent
    return ProcessState(
        timestamp=T,
        controllable={"242000:T5": 368.9},
        quality={"sulfur_mg_kg": QualityReading(
            metric="sulfur_mg_kg", value=8.0, unit="mg/kg", source="lims", age_minutes=360.0,
        )},
        data_flags=flags,
    )


def orchestrator(tmp_path, state):
    return Orchestrator(
        state_builder=FixedBuilder(state),
        artifacts_dir=tmp_path,
        whitelist=WHITELIST,
        constraints_cfg=copy.deepcopy(load_constraints()),
    )


def test_an_absent_lever_is_a_refusal_that_names_it(tmp_path):
    rec = orchestrator(tmp_path, make_state(["242000:T5"])).run_cycle(T)
    assert rec.refuse is True
    assert rec.outcome == "refuse"
    assert "242000:T5" in rec.refuse_reason
    assert "выгрузке" in rec.refuse_reason


def test_an_absent_context_tag_is_not_a_refusal(tmp_path):
    rec = orchestrator(tmp_path, make_state(["242000:Q21"])).run_cycle(T)
    assert rec.outcome != "refuse" or "242000:Q21" not in (rec.refuse_reason or "")


# ------------------------------------------------------------------ end to end

REQUIRED = ("quality_baseline.json", "reliability_reference.json", "labels.parquet")
have_data = (rd.CACHE_DIR / "tags_242000.parquet").exists() and all(
    (rd.MODELS_DIR / name).exists() for name in REQUIRED
)
needs_data = pytest.mark.skipif(not have_data, reason="нет кэша или моделей — см. README")


def _model(name: str) -> dict:
    return json.loads((rd.MODELS_DIR / name).read_text(encoding="utf-8"))


def _builder_without(tag: str, monkeypatch):
    """The real state builder over a cache whose telemetry has no `tag` column."""
    from scripts.run_real import RealStateBuilder, frozen_whitelist, load_classifier

    real = rd.cache_columns
    monkeypatch.setattr(
        rd, "cache_columns",
        lambda name: [c for c in real(name) if not (name == "tags_242000" and c == tag)],
    )
    classifier = load_classifier()
    reference = _model("reliability_reference.json")
    builder = RealStateBuilder(
        _model("quality_baseline.json"),
        classifier_features=classifier.features if classifier else None,
        reliability_reference=reference,
    )
    return builder, classifier, reference, frozen_whitelist(reference)


def _real_orchestrator(builder, classifier, reference, whitelist, tmp_path):
    """Assembled exactly as `scripts.run_real.main` does, blending agent included:
    without it the sulfur limit is 10 mg/kg instead of the tank's budget."""
    from neftecode.agents import QualityAgentBaseline, ReliabilityAgentBaseline
    from scripts.run_real import load_blending, load_economics

    return Orchestrator(
        quality_agent=QualityAgentBaseline(_model("quality_baseline.json"), classifier=classifier),
        reliability_agent=ReliabilityAgentBaseline(reference),
        state_builder=builder,
        artifacts_dir=tmp_path,
        whitelist=whitelist,
        constraints_cfg=load_constraints(),
        blending_agent=load_blending(),
        economics=load_economics(),
    )


@needs_data
def test_an_export_without_a_lever_is_refused_by_name(tmp_path, monkeypatch):
    builder, classifier, reference, whitelist = _builder_without("P13", monkeypatch)
    state = builder.build(T)
    assert state.data_flags["absent_tags"] == ["242000:P13"]
    rec = _real_orchestrator(builder, classifier, reference, whitelist, tmp_path).run_cycle(T)
    assert rec.outcome == "refuse"
    assert "242000:P13" in rec.refuse_reason


@needs_data
def test_an_export_without_a_context_tag_still_decides(tmp_path, monkeypatch):
    builder, classifier, reference, whitelist = _builder_without("Q21", monkeypatch)
    state = builder.build(T)
    assert "242000:Q21" in state.data_flags["absent_tags"]
    rec = _real_orchestrator(builder, classifier, reference, whitelist, tmp_path).run_cycle(T)
    assert rec.outcome in ("recommend", "hold")
    # The classifier needs Q21, so it steps aside and the interval estimate decides.
    quality = rec.expected_effect["quality"]
    assert (quality.get("details") or {}).get("risk_source") == "interval"
