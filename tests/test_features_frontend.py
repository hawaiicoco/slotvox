"""log_mel frontend contracts: shapes, floor, monotonicity, rejection."""

import numpy as np
import pytest

from slotvox.config import FeatureConfig
from slotvox.errors import ValidationError
from slotvox.features.frontend import LOG_FLOOR, log_mel, log_mel_frames


def sine(n=1600, freq=440.0, amp=0.8, rate=16000):
    t = np.arange(n, dtype=np.float64) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_shape_matches_frame_count():
    config = FeatureConfig()
    feats = log_mel(sine(16000), config)
    assert feats.shape == (98, 64)
    assert feats.shape[0] == log_mel_frames(16000, config)


def test_short_signal_yields_empty_features():
    feats = log_mel(sine(100), FeatureConfig())
    assert feats.shape == (0, 64)


def test_silence_hits_log_floor():
    feats = log_mel(np.zeros(1600, dtype=np.float32), FeatureConfig())
    assert feats.shape[0] > 0
    assert np.allclose(feats, np.log(LOG_FLOOR), rtol=1e-6)


def test_louder_signal_has_elementwise_larger_features():
    config = FeatureConfig()
    loud = log_mel(sine(amp=0.8), config)
    quiet = log_mel(sine(amp=0.4), config)
    assert np.all(loud >= quiet)
    assert np.any(loud > quiet)


def test_output_is_float32_and_deterministic():
    config = FeatureConfig(n_mels=8)
    samples = sine(2000)
    feats = log_mel(samples, config)
    assert feats.dtype == np.float32
    assert np.array_equal(feats, log_mel(samples, config))


def test_invalid_inputs_rejected():
    with pytest.raises(ValidationError, match="1-D"):
        log_mel(np.zeros((2, 4), dtype=np.float32), FeatureConfig())
    with pytest.raises(ValidationError, match="finite"):
        log_mel(np.array([np.nan] * 800, dtype=np.float32), FeatureConfig())
