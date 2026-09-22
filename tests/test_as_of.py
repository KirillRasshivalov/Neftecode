"""The as-of guarantee: at a moment `t` the state contains only what existed at `t`.

The data layer is the only module that reads the series, so it is the one place where
information from the future can get in — and the one place that gets a test of its own. The strongest
check here is `test_truncating_the_future_changes_nothing`: the state built over the whole
cache must equal the state built over a cache that physically ends at `t`. If anything
read forward, the two would differ.

Needs the parquet cache and the fitted models, so the whole module skips without them —
see README, steps 1 and 2.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # `scripts` lives at the root

from scripts import realdata as rd  # noqa: E402
from scripts.analysis.quality_baseline_fit import ewma_after  # noqa: E402

REQUIRED = ("quality_baseline.json", "reliability_reference.json", "labels.parquet")
missing = [name for name in REQUIRED if not (rd.MODELS_DIR / name).exists()]
if not (rd.CACHE_DIR / "tags_242000.parquet").exists():
    missing.append("data/cache/tags_242000.parquet")
pytestmark = pytest.mark.skipif(bool(missing), reason=f"нет: {', '.join(missing)}")

#: A moment inside the held-out region with the unit running and a lab result behind it.
T = datetime(2026, 7, 18, 16, 0)


def _model(name: str) -> dict:
    return json.loads((rd.MODELS_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def builder():
    from scripts.run_real import RealStateBuilder

    classifier = rd.MODELS_DIR / "breach_classifier.json"
    features = None
    if classifier.exists():
        report = json.loads(classifier.read_text(encoding="utf-8"))
        features = (report.get("linear_model") or {}).get("features")
    return RealStateBuilder(
        _model("quality_baseline.json"),
        classifier_features=features,
        reliability_reference=_model("reliability_reference.json"),
    )


@pytest.fixture(scope="module")
def state(builder):
    return builder.build(T)


# ------------------------------------------------------------------ timestamps


def test_nothing_in_the_state_is_dated_after_the_decision_time(state):
    t = pd.Timestamp(T)
    assert pd.Timestamp(state.timestamp) == t
    assert pd.Timestamp(state.data_flags["telemetry_at"]) <= t

    for reading in state.quality.values():
        assert pd.Timestamp(reading.measured_at) <= t

    pak = state.data_flags.get("pak_sulfur") or {}
    if pak:
        assert pd.Timestamp(pak["measured_at"]) <= t

    catalyst = state.data_flags.get("catalyst") or {}
    for key in ("cycle_start", "restart"):
        if catalyst.get(key):
            assert pd.Timestamp(catalyst[key]) <= t


def test_a_lab_result_is_visible_only_after_the_reporting_delay(state):
    reading = state.quality["sulfur_mg_kg"]
    assert reading.source == "lims"
    assert pd.Timestamp(reading.measured_at) + rd.LIMS_DELAY <= pd.Timestamp(T)


def test_the_age_of_every_analysis_matches_its_timestamp(state):
    t = pd.Timestamp(T)
    for reading in state.quality.values():
        expected = (t - pd.Timestamp(reading.measured_at)).total_seconds() / 60.0
        assert reading.age_minutes == pytest.approx(expected, abs=0.2)

    pak = state.data_flags.get("pak_sulfur") or {}
    if pak:
        expected = (t - pd.Timestamp(pak["measured_at"])).total_seconds() / 60.0
        assert pak["age_minutes"] == pytest.approx(expected, abs=0.2)


# ------------------------------------------------------------------ the real probe


def test_truncating_the_future_changes_nothing(builder, state, monkeypatch):
    """Rebuild over a cache that ends at `t`. Every number must come out the same."""
    from scripts.run_real import RealStateBuilder

    t = pd.Timestamp(T)
    original = rd.read_cache

    def truncated(name: str, columns: list[str] | None = None) -> pd.DataFrame:
        frame = original(name, columns)
        for column in ("date", "timestamp"):
            if column in frame.columns:
                return frame[frame[column] <= t].copy()
        return frame

    monkeypatch.setattr(rd, "read_cache", truncated)
    blind = RealStateBuilder(
        _model("quality_baseline.json"),
        classifier_features=list(builder.windows.columns) if builder.windows is not None else None,
        reliability_reference=_model("reliability_reference.json"),
    )
    monkeypatch.undo()

    blind_state = blind.build(T)
    assert blind_state.controllable == state.controllable
    assert blind_state.feature_windows == state.feature_windows
    assert blind_state.data_flags["telemetry_at"] == state.data_flags["telemetry_at"]
    assert blind_state.data_flags["pak_sulfur"] == state.data_flags["pak_sulfur"]
    assert blind_state.data_flags["lab_sulfur_ewma"] == state.data_flags["lab_sulfur_ewma"]
    assert blind_state.data_flags["catalyst"] == state.data_flags["catalyst"]


def test_the_same_moment_twice_gives_the_same_state(builder, state):
    assert builder.build(T).model_dump(mode="json") == state.model_dump(mode="json")


# ------------------------------------------------------------------ smoothing


def test_the_smoothed_lab_level_does_not_look_forward():
    """The smoothing is causal: dropping later results cannot change an earlier value."""
    values = [9.0, 11.0, 8.0, 12.0, 7.0, 10.0]
    full = ewma_after(pd.Series(values).to_numpy(), 0.3, 9.0)
    short = ewma_after(pd.Series(values[:3]).to_numpy(), 0.3, 9.0)
    assert list(full[:3]) == pytest.approx(list(short))
