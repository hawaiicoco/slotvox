"""Log-mel front end: samples in, ``[n_frames, n_mels]`` log-mel out."""

from __future__ import annotations

import numpy as np

from slotvox.audio.framing import frame_count, frame_signal, window
from slotvox.config.features import FeatureConfig
from slotvox.errors import ValidationError
from slotvox.features.mel import mel_filterbank
from slotvox.features.spectrum import stft_power

LOG_FLOOR = 1e-10


def log_mel(samples: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """Compute log-mel features for a 1-D float signal.

    Pipeline: frame -> window -> power spectrum -> mel filterbank ->
    ``log(max(mel, LOG_FLOOR))``. Output is float32 of shape
    ``[n_frames, n_mels]``; a signal shorter than one frame yields an empty
    ``[0, n_mels]`` array (documented boundary, never an error).
    """
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim != 1:
        raise ValidationError(f"log_mel expects a 1-D signal, got ndim {array.ndim}")
    if array.size and not bool(np.all(np.isfinite(array))):
        raise ValidationError("log_mel requires finite samples")
    frames = frame_signal(array, config.frame_length, config.hop_length)
    if frames.shape[0] == 0:
        return np.zeros((0, config.n_mels), dtype=np.float32)
    windowed = frames * window(config.window, config.frame_length)
    power = stft_power(windowed, config.n_fft)
    banks = mel_filterbank(
        config.sample_rate, config.n_fft, config.n_mels, config.fmin, config.fmax
    )
    mel = power @ banks.T
    return np.log(np.maximum(mel, LOG_FLOOR)).astype(np.float32)


def log_mel_frames(n_samples: int, config: FeatureConfig) -> int:
    """Number of log-mel frames a signal of ``n_samples`` produces."""
    return frame_count(n_samples, config.frame_length, config.hop_length)
