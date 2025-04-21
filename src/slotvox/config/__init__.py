"""Validated, versioned configuration records with canonical JSON roundtrip."""

from slotvox.config.base import (
    CONFIG_SCHEMA_VERSION,
    Config,
    config_from_dict,
    config_from_json,
    config_to_json,
    register_config,
    registered_kinds,
)
from slotvox.config.features import WINDOWS, FeatureConfig
from slotvox.config.generation import SPLITS, GenerationConfig
from slotvox.config.streaming import StreamConfig
from slotvox.config.training import ENCODERS, TrainConfig

__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "Config",
    "ENCODERS",
    "FeatureConfig",
    "GenerationConfig",
    "SPLITS",
    "StreamConfig",
    "TrainConfig",
    "WINDOWS",
    "config_from_dict",
    "config_from_json",
    "config_to_json",
    "register_config",
    "registered_kinds",
]
