"""render_utterance contracts: boundaries by construction, determinism."""

import numpy as np
import pytest

from slotvox.errors import ValidationError
from slotvox.synth.mix import fade_edges
from slotvox.synth.segments import render_segment
from slotvox.synth.utterance import SegmentPlan, render_utterance

RATE = 16000
PLANS = [
    SegmentPlan("intent:greet", "tone", 100, {"freq": 440.0}),
    SegmentPlan("slot:name", "chirp", 80, {"f_start": 800.0, "f_end": 1600.0}),
]


def test_boundaries_are_exact_by_construction():
    utt = render_utterance(PLANS, RATE, gap_ms=20)
    gap = 320
    first, second = utt.segments
    assert (first.start_sample, first.end_sample) == (gap, gap + 1600)
    assert (second.start_sample, second.end_sample) == (gap + 1600 + gap, gap + 1600 + gap + 1280)
    assert utt.n_samples == 3 * gap + 1600 + 1280
    assert utt.snr_db is None


def test_clean_segment_view_matches_direct_render():
    utt = render_utterance(PLANS, RATE)
    expected = fade_edges(render_segment("tone", {"freq": 440.0}, 1600, RATE), RATE)
    assert np.array_equal(utt.segment_view(utt.segments[0]), expected)
    assert np.all(utt.samples[:320] == 0.0)  # leading gap is silent


def test_render_is_deterministic_per_seed():
    a = render_utterance(PLANS, RATE, snr_db=10.0, seed=7)
    b = render_utterance(PLANS, RATE, snr_db=10.0, seed=7)
    c = render_utterance(PLANS, RATE, snr_db=10.0, seed=8)
    assert np.array_equal(a.samples, b.samples)
    assert not np.array_equal(a.samples, c.samples)
    assert a.snr_db == 10.0


def test_invalid_arguments_rejected():
    with pytest.raises(ValidationError):
        render_utterance([], RATE)
    with pytest.raises(ValidationError):
        render_utterance(["not-a-plan"], RATE)
    with pytest.raises(ValidationError):
        render_utterance(PLANS, RATE, gap_ms=-1)
    with pytest.raises(ValidationError):
        render_utterance(PLANS, RATE, snr_db=float("inf"))
    with pytest.raises(ValidationError):
        render_utterance(PLANS, RATE, seed=-5)
    with pytest.raises(ValidationError):
        render_utterance([SegmentPlan("x", "tone", 0, {"freq": 440.0})], RATE)


def test_plan_validation():
    with pytest.raises(ValidationError):
        SegmentPlan("", "tone", 100, {"freq": 440.0})
    with pytest.raises(ValidationError):
        SegmentPlan("x", "noise", 100, {})
    with pytest.raises(ValidationError):
        SegmentPlan("x", "tone", 60001, {"freq": 440.0})
