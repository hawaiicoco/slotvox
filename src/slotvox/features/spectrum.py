"""Power spectrum of windowed frames."""

from __future__ import annotations

import numpy as np

from slotvox.errors import ValidationError


def stft_power(frames: np.ndarray, n_fft: int) -> np.ndarray:
    """Magnitude-squared rFFT of each frame, zero-padded to ``n_fft``.

    Input shape ``[..., frame_length]`` with ``frame_length <= n_fft``;
    output shape ``[..., n_fft // 2 + 1]``. Padding is the only adjustment:
    frames longer than ``n_fft`` are rejected, never truncated. The
    transform is deterministic: identical inputs give bit-identical outputs.
    """
    array = np.asarray(frames, dtype=np.float64)
    if array.ndim == 0:
        raise ValidationError("stft_power expects at least 1-D frames")
    if isinstance(n_fft, bool) or not isinstance(n_fft, int) or n_fft <= 0:
        raise ValidationError(f"n_fft must be a positive integer, got {n_fft!r}")
    if array.shape[-1] > n_fft:
        raise ValidationError(
            f"frame length {array.shape[-1]} exceeds n_fft {n_fft}; padding only, no truncation"
        )
    spectrum = np.fft.rfft(array, n=n_fft, axis=-1)
    return np.abs(spectrum) ** 2
