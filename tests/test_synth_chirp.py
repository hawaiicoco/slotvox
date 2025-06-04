"""Chirp builder contracts: sweep direction, boundaries, determinism."""

import numpy as np
import pytest

from slotvox.errors import ValidationError
from slotvox.synth.signals import chirp


def band_peak_hz(segment, rate):
    spectrum = np.abs(np.fft.rfft(segment.astype(np.float64)))
    return int(np.argmax(spectrum)) * rate / len(segment)


def test_chirp_sweeps_upward():
    rate = 16000
    signal = chirp(500.0, 3000.0, 8000, rate)
    early = band_peak_hz(signal[:2000], rate)
    late = band_peak_hz(signal[-2000:], rate)
    assert early < 1200.0
    assert late > 2200.0


def test_chirp_length_and_dtype():
    signal = chirp(500.0, 1000.0, 640, 16000, amplitude=0.7)
    assert signal.dtype == np.float32
    assert signal.shape == (640,)
    assert float(np.max(np.abs(signal))) <= 0.7 + 1e-6


def test_chirp_deterministic_and_empty_allowed():
    assert np.array_equal(chirp(400.0, 800.0, 256, 8000), chirp(400.0, 800.0, 256, 8000))
    assert chirp(400.0, 800.0, 0, 8000).shape == (0,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"f_start": 1000.0, "f_end": 500.0},
        {"f_start": 500.0, "f_end": 500.0},
        {"f_start": 0.0, "f_end": 500.0},
        {"f_start": 500.0, "f_end": 9000.0},
        {"f_start": 500.0, "f_end": 1000.0, "amplitude": 2.0},
    ],
)
def test_invalid_chirp_inputs_rejected(kwargs):
    base = {"f_start": 500.0, "f_end": 1000.0, "n_samples": 512, "sample_rate": 16000}
    base.update(kwargs)
    with pytest.raises(ValidationError):
        chirp(**base)
