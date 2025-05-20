"""Golden pin for the features package API surface."""

from slotvox import features

GOLDEN_ALL = [
    "LOG_FLOOR",
    "MEL_REF_HZ",
    "MEL_SCALE",
    "hz_to_mel",
    "log_mel",
    "log_mel_frames",
    "mel_filterbank",
    "mel_to_hz",
    "stft_power",
]


def test_features_api_surface_is_pinned():
    assert sorted(features.__all__) == GOLDEN_ALL
    for name in GOLDEN_ALL:
        assert hasattr(features, name)


def test_frontend_wires_config_from_config_package():
    import numpy as np

    from slotvox.config import FeatureConfig

    config = FeatureConfig(n_mels=6)
    feats = features.log_mel(np.linspace(-1, 1, 2000), config)
    assert feats.shape == (features.log_mel_frames(2000, config), 6)
