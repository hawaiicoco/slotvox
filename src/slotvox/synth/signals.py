"""Deterministic synthetic signal builders.

Every builder here produces fully synthetic audio — pure tones, chirps, and
formant-like harmonic stacks. Nothing in this module resembles or claims to
be real speech; the signals are the acoustic carriers for labels produced by
the synthetic dialogue factory.

Conventions:
- outputs are float32 mono within [-1, 1],
- builders are deterministic: no hidden randomness (noise takes an explicit
  NumPy Generator),
- sample counts are exact: ``len(out) == n_samples``.
"""

from __future__ import annotations

import numpy as np

from slotvox.errors import ValidationError

SIGNAL_DTYPE = np.float32


def _check_rate(sample_rate: int) -> None:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValidationError(f"sample_rate must be a positive integer, got {sample_rate!r}")


def _check_n(n_samples: int) -> None:
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 0:
        raise ValidationError(f"n_samples must be a non-negative integer, got {n_samples!r}")


def _check_amplitude(amplitude: float) -> None:
    if isinstance(amplitude, bool) or not isinstance(amplitude, (int, float)):
        raise ValidationError(f"amplitude must be a number, got {amplitude!r}")
    if not np.isfinite(amplitude) or not 0.0 < float(amplitude) <= 1.0:
        raise ValidationError(f"amplitude must be within (0, 1], got {amplitude!r}")


def _check_freq(freq: float, sample_rate: int) -> None:
    if isinstance(freq, bool) or not isinstance(freq, (int, float)) or not np.isfinite(freq):
        raise ValidationError(f"frequency must be a finite number, got {freq!r}")
    nyquist = sample_rate / 2
    if not 0.0 < float(freq) <= nyquist:
        raise ValidationError(f"frequency {freq} must be within (0, Nyquist={nyquist}]")


def time_base(n_samples: int, sample_rate: int) -> np.ndarray:
    """Sample times in seconds (float64)."""
    _check_n(n_samples)
    _check_rate(sample_rate)
    return np.arange(n_samples, dtype=np.float64) / sample_rate


def silence(n_samples: int) -> np.ndarray:
    """Digital silence (float32 zeros) of exact length."""
    _check_n(n_samples)
    return np.zeros(n_samples, dtype=SIGNAL_DTYPE)


def tone(
    freq: float,
    n_samples: int,
    sample_rate: int,
    *,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> np.ndarray:
    """Pure sine at ``freq`` Hz with optional amplitude and phase."""
    _check_rate(sample_rate)
    _check_n(n_samples)
    _check_freq(freq, sample_rate)
    _check_amplitude(amplitude)
    if isinstance(phase, bool) or not isinstance(phase, (int, float)) or not np.isfinite(phase):
        raise ValidationError(f"phase must be a finite number, got {phase!r}")
    t = time_base(n_samples, sample_rate)
    wave = float(amplitude) * np.sin(2.0 * np.pi * float(freq) * t + float(phase))
    return wave.astype(SIGNAL_DTYPE)
