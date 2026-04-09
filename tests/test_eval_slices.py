"""Slice summaries: grouping, goldens, and determinism."""

import pytest

from slotvox.errors import ValidationError
from slotvox.eval.metrics import TurnRecord, slice_pairs
from slotvox.eval.slices import slice_summary


def record(gi, pi, gt, pt, slices):
    return TurnRecord(gi, pi, gt, pt, slices=slice_pairs(slices))


def test_groups_and_golden_values():
    records = [
        record("a", "a", ("B-city", "O"), ("B-city", "O"), {"noise": "clean"}),
        record("a", "b", ("B-city", "O"), ("O", "O"), {"noise": "snr-15"}),
        record("b", "b", ("O",), ("O",), {"noise": "snr-15"}),
    ]
    summary = slice_summary(records)
    assert list(summary) == ["noise"]
    assert list(summary["noise"]) == ["clean", "snr-15"]
    assert summary["noise"]["clean"] == {
        "count": 1,
        "intent_accuracy": 1.0,
        "slot_f1_strict": 1.0,
        "slot_f1_partial": 1.0,
    }
    noisy = summary["noise"]["snr-15"]
    assert noisy["count"] == 2
    assert noisy["intent_accuracy"] == 0.5
    assert noisy["slot_f1_strict"] == 0.0  # one gold span missed, nothing spurious
    assert noisy["slot_f1_partial"] == 0.0


def test_multiple_kinds_sorted():
    records = [
        record("a", "a", ("O",), ("O",), {"noise": "clean", "domain": "weather"}),
        record("a", "a", ("O",), ("O",), {"domain": "music-control"}),
    ]
    summary = slice_summary(records)
    assert list(summary) == ["domain", "noise"]
    assert list(summary["domain"]) == ["music-control", "weather"]


def test_no_slices_and_empty_input():
    assert slice_summary([record("a", "a", ("O",), ("O",), {})]) == {}
    assert slice_summary([]) == {}


def test_rejections():
    with pytest.raises(ValidationError, match="TurnRecord"):
        slice_summary(["nope"])
    with pytest.raises(ValidationError, match="list/tuple"):
        slice_summary("nope")
