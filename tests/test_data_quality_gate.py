from datetime import datetime

from neftecode.domain.state import ProcessState, QualityReading
from neftecode.safety.data_quality_gate import DataQualityGate


def _state(*, age_minutes: float, pak_healthy: bool | None, source: str = "lims") -> ProcessState:
    flags: dict = {"complete": True}
    if pak_healthy is not None:
        flags["pak_sulfur"] = {"healthy": pak_healthy, "frozen": False, "age_minutes": 10.0}
    return ProcessState(
        timestamp=datetime(2026, 1, 27, 16, 0, 0),
        quality={
            "sulfur_mg_kg": QualityReading(
                metric="sulfur_mg_kg",
                value=8.5,
                unit="mg/kg",
                source=source,  # type: ignore[arg-type]
                measured_at=datetime(2026, 1, 26, 10, 0, 0),
                age_minutes=age_minutes,
            )
        },
        data_flags=flags,
    )


def test_fresh_lims_passes():
    gate = DataQualityGate()
    result = gate.check(_state(age_minutes=300, pak_healthy=False))
    assert result.ok
    assert not result.reasons


def test_stale_lims_with_healthy_pak_passes_with_warning():
    gate = DataQualityGate()
    result = gate.check(_state(age_minutes=10_000, pak_healthy=True))
    assert result.ok
    assert not result.reasons
    assert any("поточный анализатор" in w for w in result.warnings)
    assert any("ч назад" in w for w in result.warnings)


def test_stale_lims_without_pak_refuses_in_russian_hours():
    gate = DataQualityGate()
    result = gate.check(_state(age_minutes=20_880, pak_healthy=False))
    assert not result.ok
    assert any("348" in r and "ч" in r for r in result.reasons)
    assert any("30" in r for r in result.reasons)


def test_missing_lims_with_healthy_pak_passes():
    gate = DataQualityGate()
    state = ProcessState(
        timestamp=datetime(2026, 1, 27, 16, 0, 0),
        quality={},
        data_flags={"complete": True, "pak_sulfur": {"healthy": True}},
    )
    result = gate.check(state)
    assert result.ok
    assert result.warnings


def test_missing_lims_and_pak_refuses():
    gate = DataQualityGate()
    state = ProcessState(
        timestamp=datetime(2026, 1, 27, 16, 0, 0),
        quality={},
        data_flags={"complete": True, "pak_sulfur": {"healthy": False}},
    )
    result = gate.check(state)
    assert not result.ok
