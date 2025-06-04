"""Tone/silence builder contracts: frequency, amplitude, determinism."""

import numpy as np
import pytest

from slotvox.errors import ValidationError
from slotvox.synth.signals import silence, time_base, tone


def test_tone_peak_frequency_matches_fft():
    rate = 16000
    signal = tone(440.0, 4096, rate).astype(np.float64)
    spectrum = np.abs(np.fft.rfft(signal))
    peak_hz = int(np.argmax(spectrum)) * rate / 4096
    assert abs(peak_hz - 440.0) < rate / 4096


def test_tone_amplitude_dtype_and_length():
    signal = tone(440.0, 1000, 16000, amplitude=0.5)
    assert signal.dtype == np.float32
    assert signal.shape == (1000,)
    assert float(np.max(np.abs(signal))) == pytest.approx(0.5, abs=0.01)


def test_tone_phase_starts_at_sin_of_phase():
    first = tone(440.0, 8, 16000, phase=np.pi / 2)[0]
    assert float(first) == pytest.approx(1.0, abs=1e-5)


def test_tone_is_deterministic():
    assert np.array_equal(tone(300.0, 64, 8000), tone(300.0, 64, 8000))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"freq": 0.0},
        {"freq": -5.0},
        {"freq": 8001.0},
        {"amplitude": 0.0},
        {"amplitude": 1.5},
        {"n_samples": -1},
        {"n_samples": 2.0},
        {"sample_rate": 0},
        {"phase": float("nan")},
    ],
)
def test_invalid_tone_inputs_rejected(kwargs):
    base = {"freq": 440.0, "n_samples": 64, "sample_rate": 16000}
    base.update(kwargs)
    with pytest.raises(ValidationError):
        tone(**base)


def test_silence_and_time_base():
    assert np.array_equal(silence(5), np.zeros(5, dtype=np.float32))
    assert silence(0).shape == (0,)
    assert np.allclose(time_base(3, 1000), [0.0, 0.001, 0.002])
    with pytest.raises(ValidationError):
        silence(-1)
