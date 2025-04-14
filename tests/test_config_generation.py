"""GenerationConfig validation and roundtrip contracts."""

import pytest

from slotvox.config.generation import GenerationConfig
from slotvox.errors import ConfigError


def test_defaults_round_trip():
    config = GenerationConfig()
    assert GenerationConfig.from_dict(config.to_dict()) == config
    assert set(config.counts) == {"train", "dev", "test"}


def test_unknown_split_rejected():
    with pytest.raises(ConfigError, match="unknown splits"):
        GenerationConfig(counts={"train": 4, "validation": 2})


def test_negative_or_empty_counts_rejected():
    with pytest.raises(ConfigError):
        GenerationConfig(counts={"train": -1})
    with pytest.raises(ConfigError, match="at least one"):
        GenerationConfig(counts={"train": 0, "dev": 0})


def test_noise_levels_normalize_to_tuple_and_round_trip():
    config = GenerationConfig(noise_snr_db=[25.0, 10.0])
    assert config.noise_snr_db == (25.0, 10.0)
    assert GenerationConfig.from_dict(config.to_dict()) == config


def test_invalid_noise_levels_rejected():
    with pytest.raises(ConfigError):
        GenerationConfig(noise_snr_db=[])
    with pytest.raises(ConfigError):
        GenerationConfig(noise_snr_db="loud")
    with pytest.raises(ConfigError):
        GenerationConfig(noise_snr_db=(-3.0,))


def test_speaker_seed_and_domain_bounds():
    with pytest.raises(ConfigError):
        GenerationConfig(n_speakers=0)
    with pytest.raises(ConfigError):
        GenerationConfig(n_speakers=65)
    with pytest.raises(ConfigError):
        GenerationConfig(seed=-1)
    with pytest.raises(ConfigError):
        GenerationConfig(domain="")
