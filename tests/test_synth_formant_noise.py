"""Formant-pattern and noise builder contracts."""

import numpy as np
import pytest

from slotvox.errors import ValidationError
from slotvox.synth.signals import MAX_FORMANTS, formant_pattern, noise
from slotvox.util.seed import make_rng


def spectral_peaks(signal, rate, count):
    spectrum = np.abs(np.fft.rfft(signal.astype(np.float64)))
    interior = np.flatnonzero((spectrum[1:-1] > spectrum[:-2]) & (spectrum[1:-1] >= spectrum[2:]))
    ranked = interior[np.argsort(spectrum[interior + 1])[::-1]]
    return [float(rate * (index + 1) / len(signal)) for index in ranked[:count]]


def test_formant_peaks_match_requested_frequencies():
    rate = 16000
    signal = formant_pattern([400.0, 1200.0, 2200.0], 4096, rate)
    peaks = sorted(spectral_peaks(signal, rate, 3))
    assert peaks == pytest.approx([400.0, 1200.0, 2200.0], abs=25.0)


def test_formant_amplitude_peak_normalized():
    signal = formant_pattern([300.0, 900.0], 2048, 16000, amplitude=0.6)
    assert float(np.max(np.abs(signal))) == pytest.approx(0.6, abs=1e-5)


@pytest.mark.parametrize(
    "formants",
    [
        [],
        [400.0] * (MAX_FORMANTS + 1),
        [900.0, 400.0],
        [400.0, 400.0],
        [0.0, 400.0],
        [400.0, 9000.0],
        "400",
    ],
)
def test_invalid_formants_rejected(formants):
    with pytest.raises(ValidationError):
        formant_pattern(formants, 512, 16000)


def test_noise_is_deterministic_per_seed():
    first = noise(1000, 16000, amplitude=0.5, rng=make_rng(3))
    second = noise(1000, 16000, amplitude=0.5, rng=make_rng(3))
    other = noise(1000, 16000, amplitude=0.5, rng=make_rng(4))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_noise_stays_bounded_and_rejects_bad_rng():
    signal = noise(4000, 16000, amplitude=1.0, rng=make_rng(5))
    assert float(np.max(np.abs(signal))) <= 1.0
    with pytest.raises(ValidationError, match="Generator"):
        noise(16, 16000, amplitude=0.5, rng=42)
    with pytest.raises(ValidationError):
        noise(16, 16000, amplitude=0.0, rng=make_rng(1))
