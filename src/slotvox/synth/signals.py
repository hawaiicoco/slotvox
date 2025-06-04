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


def chirp(
    f_start: float,
    f_end: float,
    n_samples: int,
    sample_rate: int,
    *,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Linear frequency sweep from ``f_start`` to ``f_end`` Hz.

    The instantaneous frequency at time ``t`` (within duration ``T``) is
    ``f_start + (f_end - f_start) * t / T``, obtained from the quadratic
    phase ``2*pi*(f_start*t + (f_end - f_start)*t^2 / (2*T))``.
    """
    _check_rate(sample_rate)
    _check_n(n_samples)
    _check_freq(f_start, sample_rate)
    _check_freq(f_end, sample_rate)
    _check_amplitude(amplitude)
    if not float(f_start) < float(f_end):
        raise ValidationError(f"chirp requires f_start < f_end, got {f_start} >= {f_end}")
    if n_samples == 0:
        return np.zeros(0, dtype=SIGNAL_DTYPE)
    duration = n_samples / sample_rate
    t = time_base(n_samples, sample_rate)
    phase = (
        2.0
        * np.pi
        * (float(f_start) * t + (float(f_end) - float(f_start)) * t**2 / (2.0 * duration))
    )
    return (float(amplitude) * np.sin(phase)).astype(SIGNAL_DTYPE)


MAX_FORMANTS = 8


def formant_pattern(
    formants: list[float] | tuple[float, ...],
    n_samples: int,
    sample_rate: int,
    *,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Sum of sine "formants" with 1/k weights, peak-normalized.

    A crude formant-like stack that gives slot values a timbral signature;
    it is synthetic signaling, not a vocal-tract model. ``formants`` must be
    strictly ascending frequencies within (0, Nyquist].
    """
    _check_rate(sample_rate)
    _check_n(n_samples)
    _check_amplitude(amplitude)
    if isinstance(formants, (str, bytes)) or not isinstance(formants, (list, tuple)):
        raise ValidationError(f"formants must be a list/tuple of frequencies, got {formants!r}")
    if not 1 <= len(formants) <= MAX_FORMANTS:
        raise ValidationError(f"formants needs 1..{MAX_FORMANTS} entries, got {len(formants)}")
    previous = 0.0
    for value in formants:
        _check_freq(value, sample_rate)
        if float(value) <= previous:
            raise ValidationError(f"formants must be strictly ascending, got {list(formants)}")
        previous = float(value)
    t = time_base(n_samples, sample_rate)
    signal = np.zeros(n_samples, dtype=np.float64)
    for index, value in enumerate(formants):
        signal += np.sin(2.0 * np.pi * float(value) * t) / (index + 1)
    peak = float(np.max(np.abs(signal))) if n_samples else 0.0
    if peak > 0.0:
        signal = signal * (float(amplitude) / peak)
    return signal.astype(SIGNAL_DTYPE)


def noise(
    n_samples: int,
    sample_rate: int,
    *,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """White Gaussian noise with standard deviation ``amplitude / 3``.

    Randomness flows only through the explicit ``rng`` (see
    :func:`slotvox.util.seed.make_rng`), so results are reproducible per
    seed. Output is clipped to [-1, 1]; with ``amplitude <= 1`` clipping is
    a >3-sigma event per sample and measurably rare.
    """
    _check_rate(sample_rate)
    _check_n(n_samples)
    _check_amplitude(amplitude)
    if not isinstance(rng, np.random.Generator):
        raise ValidationError(
            f"rng must be a numpy Generator (see make_rng), got {type(rng).__name__}"
        )
    values = rng.standard_normal(n_samples) * (float(amplitude) / 3.0)
    return np.clip(values, -1.0, 1.0).astype(SIGNAL_DTYPE)
