"""FeatureConfig validation boundaries and roundtrip."""

import pytest

from slotvox.config.features import WINDOWS, FeatureConfig
from slotvox.errors import ConfigError


def test_defaults_construct_and_derive_frame_geometry():
    config = FeatureConfig()
    assert config.frame_length == 400
    assert config.hop_length == 160


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 999},
        {"sample_rate": 192001},
        {"sample_rate": 16000.0},
        {"frame_ms": 0},
        {"hop_ms": 0},
        {"hop_ms": 26},
        {"n_fft": 511},
        {"n_mels": 0},
        {"fmin": -1.0},
        {"fmin": 7600.0},
        {"fmax": 8001.0},
        {"window": "kaiser"},
    ],
)
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ConfigError):
        FeatureConfig(**kwargs)


def test_fmax_at_nyquist_boundary_accepted():
    assert FeatureConfig(fmax=8000.0).fmax == 8000.0


def test_custom_geometry_round_trip():
    config = FeatureConfig(
        sample_rate=8000,
        frame_ms=32,
        hop_ms=8,
        n_fft=256,
        n_mels=40,
        fmin=50.0,
        fmax=3900.0,
        window="hamming",
    )
    assert FeatureConfig.from_dict(config.to_dict()) == config
    assert config.window in WINDOWS
