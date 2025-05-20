"""log_mel must equal the manual composition of its public building blocks."""

import numpy as np

from slotvox.audio.framing import frame_signal, window
from slotvox.config import FeatureConfig
from slotvox.features.frontend import LOG_FLOOR, log_mel
from slotvox.features.mel import mel_filterbank
from slotvox.features.spectrum import stft_power


def test_log_mel_equals_manual_composition():
    config = FeatureConfig(n_mels=16)
    samples = (np.linspace(0, 1, 4000) * np.sin(np.linspace(0, 60, 4000))).astype(np.float64)
    frames = frame_signal(samples, config.frame_length, config.hop_length)
    windowed = frames * window(config.window, config.frame_length)
    power = stft_power(windowed, config.n_fft)
    banks = mel_filterbank(
        config.sample_rate, config.n_fft, config.n_mels, config.fmin, config.fmax
    )
    expected = np.log(np.maximum(power @ banks.T, LOG_FLOOR)).astype(np.float32)
    assert np.array_equal(log_mel(samples, config), expected)
