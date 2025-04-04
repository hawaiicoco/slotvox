"""Contracts for the config kind registry and JSON helpers."""

from dataclasses import dataclass
from typing import ClassVar

import pytest

from slotvox.config.base import (
    Config,
    config_from_dict,
    config_from_json,
    config_to_json,
    register_config,
    registered_kinds,
)
from slotvox.errors import ConfigError


@register_config
@dataclass(frozen=True)
class Registered(Config):
    kind: ClassVar[str] = "demo-registered"

    value: int = 3


def test_config_from_dict_dispatches_on_kind():
    assert config_from_dict(Registered().to_dict()) == Registered(value=3)


def test_unknown_kind_rejected_listing_registered():
    with pytest.raises(ConfigError, match="unknown config kind"):
        config_from_dict({"kind": "nope", "schema_version": 1})


def test_duplicate_kind_registration_rejected():
    with pytest.raises(ConfigError, match="duplicate config kind"):

        @register_config
        @dataclass(frozen=True)
        class Impostor(Config):
            kind: ClassVar[str] = "demo-registered"


def test_base_class_without_concrete_kind_rejected():
    with pytest.raises(ConfigError, match="concrete kind"):

        @register_config
        @dataclass(frozen=True)
        class Anonymous(Config):
            value: int = 1


def test_json_helpers_round_trip():
    config = Registered(value=5)
    assert config_from_json(config_to_json(config)) == config


def test_registered_kinds_is_sorted_and_contains_demo():
    kinds = registered_kinds()
    assert "demo-registered" in kinds
    assert list(kinds) == sorted(kinds)
