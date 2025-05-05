"""Mel scale conversion contracts, both directions."""

import numpy as np
import pytest

from slotvox.errors import ValidationError
from slotvox.features.mel import MEL_REF_HZ, hz_to_mel, mel_to_hz


def test_zero_maps_to_zero():
    assert hz_to_mel(0.0) == 0.0
    assert mel_to_hz(0.0) == 0.0


def test_reference_frequency_golden():
    assert hz_to_mel(MEL_REF_HZ) == pytest.approx(781.1728387480312, rel=1e-14)


def test_round_trip_hz_to_mel_and_back():
    for hz in (0.0, 100.0, 700.0, 1000.0, 8000.0):
        assert mel_to_hz(hz_to_mel(hz)) == pytest.approx(hz, rel=1e-12)


def test_round_trip_mel_to_hz_and_back():
    for mel in (0.0, 500.0, 2595.0):
        assert hz_to_mel(mel_to_hz(mel)) == pytest.approx(mel, rel=1e-12)


def test_monotonic_and_elementwise_for_arrays():
    hz = np.array([0.0, 350.0, 700.0, 4000.0])
    mels = hz_to_mel(hz)
    assert isinstance(mels, np.ndarray)
    assert mels.shape == (4,)
    assert np.all(np.diff(mels) > 0)


def test_invalid_inputs_rejected_both_directions():
    with pytest.raises(ValidationError):
        hz_to_mel(-1.0)
    with pytest.raises(ValidationError):
        hz_to_mel(float("nan"))
    with pytest.raises(ValidationError):
        mel_to_hz(-0.5)
    with pytest.raises(ValidationError):
        mel_to_hz(float("inf"))
