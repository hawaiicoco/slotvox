"""Quality flags: honest unknowns, both-direction bounds, report stats."""

import pytest

from slotvox.errors import ValidationError
from slotvox.instructions.quality import (
    QualityFlags,
    QualityReport,
    QualityThresholds,
    canonical_answer_prefix,
    check_sample,
    check_samples,
)
from slotvox.instructions.schema import AudioRef, InstructionSample, Turn

HASH = "d4" * 32


def make(sample_id="weather-train-00000", duration=500.0, answer="意图：query-weather；槽位：无"):
    return InstructionSample(
        sample_id=sample_id,
        domain="weather",
        language="zh",
        intent="query-weather",
        turns=(
            Turn("system", text="提示"),
            Turn("user", audio=AudioRef(sample_id, HASH, duration)),
            Turn("assistant", text=answer),
        ),
        tags=("synthetic",),
    )


def test_canonical_prefix_derived_from_templates():
    assert canonical_answer_prefix("zh") == "意图："
    assert canonical_answer_prefix("en") == "Intent: "
    with pytest.raises(ValidationError, match="language"):
        canonical_answer_prefix("fr")


def test_passing_sample_has_no_issues():
    flags = check_sample(make())
    assert flags.ok
    assert flags.duration_ok and flags.answer_canonical
    assert flags.snr_ok is None  # honest unknown without sidecar data
    assert flags.issues == ()


def test_duration_bounds_both_directions():
    assert "duration-out-of-bounds" in check_sample(make(duration=50.0)).issues
    assert "duration-out-of-bounds" in check_sample(make(duration=40000.0)).issues
    assert check_sample(make(duration=200.0)).duration_ok  # inclusive bounds
    assert check_sample(make(duration=30000.0)).duration_ok


def test_snr_sidecar_rules():
    thresholds = QualityThresholds(min_snr_db=10.0)
    good = check_sample(make(), snr_db={"weather-train-00000": 15.0}, thresholds=thresholds)
    assert good.snr_ok is True and good.ok
    low = check_sample(make(), snr_db={"weather-train-00000": 5.0}, thresholds=thresholds)
    assert low.snr_ok is False and "snr-below-minimum" in low.issues
    absent = check_sample(make(), snr_db={}, thresholds=thresholds)
    assert absent.snr_ok is None and absent.ok  # missing data is not failure
    with pytest.raises(ValidationError, match="finite"):
        check_sample(make(), snr_db={"weather-train-00000": "loud"}, thresholds=thresholds)


def test_answer_canonicality():
    flags = check_sample(make(answer="the weather is nice"))
    assert not flags.answer_canonical
    assert "answer-not-canonical" in flags.issues


def test_report_aggregation_and_threshold_validation():
    report = check_samples([make(), make(sample_id="weather-train-00001", duration=1.0)])
    assert (report.passed, report.failed) == (1, 1)
    summary = report.to_dict()
    assert summary["sample_count"] == 2
    assert summary["issues"] == {"duration-out-of-bounds": 1}
    with pytest.raises(ValidationError, match="durations"):
        QualityThresholds(min_duration_ms=500.0, max_duration_ms=100.0)
    with pytest.raises(ValidationError, match="min_snr_db"):
        QualityThresholds(min_snr_db=float("nan"))
    with pytest.raises(ValidationError):
        check_samples("nope")


def test_flags_validation():
    with pytest.raises(ValidationError, match="snr_ok"):
        QualityFlags("s", True, "yes", True, ())
    with pytest.raises(ValidationError, match="issues"):
        QualityFlags("s", True, None, True, (3,))
    with pytest.raises(ValidationError, match="equal"):
        QualityReport((), 1, 0)
