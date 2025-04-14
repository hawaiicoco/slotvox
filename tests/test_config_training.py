"""TrainConfig validation and roundtrip contracts."""

import pytest

from slotvox.config.training import TrainConfig
from slotvox.errors import ConfigError


def test_defaults_round_trip():
    config = TrainConfig()
    assert TrainConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"encoder": "lstm"},
        {"hidden_size": 2},
        {"hidden_size": 2048},
        {"layers": 0},
        {"layers": 5},
        {"dropout": 0.95},
        {"epochs": 0},
        {"batch_size": 0},
        {"learning_rate": 0.0},
        {"learning_rate": 2.0},
        {"grad_clip_norm": 0.0},
    ],
)
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ConfigError):
        TrainConfig(**kwargs)


def test_zero_total_loss_weight_rejected():
    with pytest.raises(ConfigError, match="positive total loss weight"):
        TrainConfig(intent_loss_weight=0.0, slot_loss_weight=0.0)


def test_loss_weight_boundaries_accepted():
    assert TrainConfig(intent_loss_weight=0.0, slot_loss_weight=1.0).epochs == 10
    assert TrainConfig(dropout=0.9).dropout == 0.9
    assert TrainConfig(hidden_size=4).hidden_size == 4
