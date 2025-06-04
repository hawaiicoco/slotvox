"""apply_snr and fade_edges contracts, measured in both directions."""

import numpy as np
import pytest

from slotvox.errors import ValidationError
from slotvox.synth.mix import apply_snr, fade_edges
from slotvox.synth.signals import noise, tone
from slotvox.util.seed import make_rng

RATE = 16000


def test_measured_snr_matches_target():
    signal = tone(440.0, RATE, RATE, amplitude=0.5)
    overlay = noise(RATE, RATE, amplitude=0.5, rng=make_rng(1))
    mixed = apply_snr(signal, overlay, 10.0)
    residue = mixed.astype(np.float64) - signal.astype(np.float64)
    signal_power = float(np.mean(signal.astype(np.float64) ** 2))
    noise_power = float(np.mean(residue**2))
    measured = 10.0 * np.log10(signal_power / noise_power)
    assert measured == pytest.approx(10.0, abs=0.05)


def test_zero_power_noise_returns_signal():
    signal = tone(440.0, 800, RATE, amplitude=0.5)
    assert np.array_equal(apply_snr(signal, np.zeros(800, dtype=np.float32), 20.0), signal)


def test_silent_signal_and_mismatched_lengths_rejected():
    overlay = noise(800, RATE, amplitude=0.5, rng=make_rng(2))
    with pytest.raises(ValidationError, match="non-silent"):
        apply_snr(np.zeros(800, dtype=np.float32), overlay, 10.0)
    with pytest.raises(ValidationError, match="equal length"):
        apply_snr(tone(440.0, 801, RATE), overlay, 10.0)
    with pytest.raises(ValidationError, match="snr_db"):
        apply_snr(tone(440.0, 800, RATE), overlay, float("nan"))


def test_fade_zeroes_first_and_last_sample_only_at_edges():
    signal = np.ones(1000, dtype=np.float32)
    faded = fade_edges(signal, RATE, attack_ms=5, release_ms=5)
    assert faded[0] == 0.0
    assert faded[-1] == 0.0
    assert faded[500] == pytest.approx(1.0)
    assert faded.shape == signal.shape


def test_fade_zero_ms_is_identity_and_short_signals_rejected():
    signal = np.ones(64, dtype=np.float32)
    assert np.array_equal(fade_edges(signal, RATE, attack_ms=0, release_ms=0), signal)
    with pytest.raises(ValidationError, match="too short"):
        fade_edges(signal, RATE, attack_ms=500, release_ms=500)
    with pytest.raises(ValidationError):
        fade_edges(signal, RATE, attack_ms=-1)
