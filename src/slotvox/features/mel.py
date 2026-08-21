"""Mel scale conversions.

The mel scale used here is the HTK variant ``mel = 2595 * log10(1 + hz/700)``,
with :func:`mel_to_hz` as its exact inverse. Conversions are elementwise for
arrays and reject negative or non-finite inputs in both directions.
"""

from __future__ import annotations

import numpy as np

from slotvox.errors import ValidationError

MEL_REF_HZ = 700.0
MEL_SCALE = 2595.0


def hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    """Convert Hz to mel (HTK formula); elementwise for arrays."""
    array = np.asarray(hz, dtype=np.float64)
    if np.any(array < 0) or not np.all(np.isfinite(array)):
        raise ValidationError("hz_to_mel requires finite, non-negative frequencies")
    result = MEL_SCALE * np.log10(1.0 + array / MEL_REF_HZ)
    return float(result) if result.ndim == 0 else result


def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    """Convert mel to Hz (exact inverse of :func:`hz_to_mel`)."""
    array = np.asarray(mel, dtype=np.float64)
    if np.any(array < 0) or not np.all(np.isfinite(array)):
        raise ValidationError("mel_to_hz requires finite, non-negative mel values")
    result = MEL_REF_HZ * (10.0 ** (array / MEL_SCALE) - 1.0)
    return float(result) if result.ndim == 0 else result


def mel_filterbank(
    sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> np.ndarray:
    """Triangular mel filterbank matrix of shape ``[n_mels, n_fft // 2 + 1]``.

    Each row is a triangle rising from ``low`` to ``center`` and falling to
    ``high``, with mel-equidistant edges sampled at rFFT bin centers;
    weights are peak-normalized to [0, 1]. A filter that captures no bin at
    all (an ``n_fft`` too coarse for the requested density) is rejected
    instead of silently contributing an all-zero row.
    """
    _check_bank_params(sample_rate, n_fft, n_mels, fmin, fmax)
    n_freqs = n_fft // 2 + 1
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = np.asarray(mel_to_hz(mel_points))
    bin_hz = np.arange(n_freqs, dtype=np.float64) * (sample_rate / n_fft)
    banks = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for m in range(n_mels):
        low, center, high = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        rising = (bin_hz - low) / (center - low)
        falling = (high - bin_hz) / (high - center)
        banks[m] = np.clip(np.minimum(rising, falling), 0.0, 1.0)
        if not banks[m].any():
            raise ValidationError(
                f"mel filter {m} ({low:.1f}-{high:.1f} Hz) captures no FFT bins; "
                "reduce n_mels or increase n_fft"
            )
    return banks


def _check_bank_params(sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> None:
    for name, value in (("sample_rate", sample_rate), ("n_fft", n_fft), ("n_mels", n_mels)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValidationError(f"{name} must be a positive integer, got {value!r}")
    if n_fft < 2 or n_fft & (n_fft - 1):
        raise ValidationError(f"n_fft must be a power of two >= 2, got {n_fft}")
    if not 0.0 <= fmin < fmax <= sample_rate / 2:
        raise ValidationError(
            f"require 0 <= fmin < fmax <= Nyquist ({sample_rate / 2}), got [{fmin}, {fmax}]"
        )
