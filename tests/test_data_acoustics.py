"""Token->acoustics contracts: determinism, bands, speaker groups."""

import pytest

from slotvox.data.acoustics import (
    FILLER_BAND,
    MAX_SPEAKER_SHIFT,
    SLOT_BANDS,
    SLOT_BASE,
    SLOT_STEP,
    speaker_shift,
    token_segment_plan,
)
from slotvox.errors import ValidationError


def test_identical_inputs_give_identical_plans():
    a = token_segment_plan("weather", "北", "city")
    b = token_segment_plan("weather", "北", "city")
    assert a == b


def test_filler_tokens_are_low_band_tones():
    plan = token_segment_plan("weather", "天", None)
    assert plan.kind == "tone"
    freq = plan.params["freq"]
    low = FILLER_BAND[0] * (1 - MAX_SPEAKER_SHIFT) - 1
    high = FILLER_BAND[1] * (1 + MAX_SPEAKER_SHIFT) + 1
    assert low <= freq <= high


def test_slot_tokens_are_ascending_formant_pairs_in_band():
    plan = token_segment_plan("weather", "北", "city")
    assert plan.kind == "formant"
    first, second = plan.params["formants"]
    assert first < second
    assert first >= SLOT_BASE * (1 - MAX_SPEAKER_SHIFT)
    assert first < (SLOT_BASE + SLOT_BANDS * SLOT_STEP) * (1 + MAX_SPEAKER_SHIFT)


def test_speaker_shift_is_bounded_deterministic_and_varied():
    assert speaker_shift(0) == speaker_shift(0)
    assert speaker_shift(0) != speaker_shift(1)
    for speaker in range(8):
        assert 1 - MAX_SPEAKER_SHIFT <= speaker_shift(speaker) <= 1 + MAX_SPEAKER_SHIFT


def test_speaker_group_changes_acoustics_but_not_labels():
    a = token_segment_plan("weather", "北", "city", speaker_id=0)
    b = token_segment_plan("weather", "北", "city", speaker_id=1)
    assert a.label == b.label
    assert a.params != b.params


@pytest.mark.parametrize(
    "call",
    [
        lambda: speaker_shift(-1),
        lambda: speaker_shift(1.5),
        lambda: token_segment_plan("", "北", None),
        lambda: token_segment_plan("weather", "", None),
        lambda: token_segment_plan("weather", "a b", None),
        lambda: token_segment_plan("weather", "北", "city", duration_ms=5),
        lambda: token_segment_plan("weather", "北", "city", duration_ms=2001),
        lambda: token_segment_plan("weather", "北", "Bad Slot"),
    ],
)
def test_invalid_inputs_rejected(call):
    with pytest.raises(ValidationError):
        call()
