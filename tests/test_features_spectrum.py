"""stft_power contracts: shapes, known spectra, rejection, determinism."""

import numpy as np
import pytest

from slotvox.errors import ValidationError
from slotvox.features.spectrum import stft_power


def test_silence_has_zero_power():
    power = stft_power(np.zeros((2, 8)), 16)
    assert power.shape == (2, 9)
    assert np.all(power == 0.0)


def test_dc_signal_concentrates_in_bin_zero():
    power = stft_power(np.ones((1, 8)), 8)[0]
    assert power[0] == pytest.approx(64.0)
    assert np.allclose(power[1:], 0.0, atol=1e-9)


def test_zero_padding_of_short_frames_peaks_at_nyquist_bin():
    frame = np.array([1.0, -1.0, 1.0, -1.0])
    power = stft_power(frame, 8)
    assert power.shape == (5,)
    assert power[4] == pytest.approx(16.0)
    assert int(np.argmax(power)) == 4


def test_padding_only_rule_rejects_long_frames():
    with pytest.raises(ValidationError, match="exceeds n_fft"):
        stft_power(np.zeros((2, 8)), 4)


def test_invalid_n_fft_rejected():
    for bad in (0, -2, True, 8.0):
        with pytest.raises(ValidationError):
            stft_power(np.zeros((1, 4)), bad)


def test_scalar_input_rejected():
    with pytest.raises(ValidationError, match="1-D"):
        stft_power(np.float64(1.0), 8)


def test_deterministic_bit_identical_outputs():
    frames = np.linspace(-1, 1, 24).reshape(3, 8)
    assert np.array_equal(stft_power(frames, 16), stft_power(frames, 16))
