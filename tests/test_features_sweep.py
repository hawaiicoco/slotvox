"""Small-case boundary sweeps across mel counts and windows."""

import numpy as np

from slotvox.audio.framing import WINDOW_NAMES
from slotvox.config import FeatureConfig
from slotvox.features.frontend import log_mel, log_mel_frames


def test_mel_count_sweep():
    samples = (np.sin(np.linspace(0, 50, 2400)) * np.linspace(0, 1, 2400)).astype(np.float64)
    for n_mels in range(1, 9):
        config = FeatureConfig(sample_rate=8000, n_fft=256, n_mels=n_mels, fmin=0.0, fmax=4000.0)
        feats = log_mel(samples, config)
        assert feats.shape == (log_mel_frames(2400, config), n_mels)
        assert np.all(np.isfinite(feats))


def test_all_windows_produce_finite_features():
    samples = np.linspace(-1, 1, 3000)
    for name in WINDOW_NAMES:
        feats = log_mel(samples, FeatureConfig(n_mels=8, window=name))
        assert feats.shape[1] == 8
        assert np.all(np.isfinite(feats))


def test_minimum_viable_config():
    config = FeatureConfig(
        sample_rate=1000, frame_ms=10, hop_ms=10, n_fft=16, n_mels=1, fmin=0.0, fmax=500.0
    )
    feats = log_mel(np.ones(40, dtype=np.float64), config)
    assert feats.shape == (4, 1)
    assert np.all(np.isfinite(feats))
