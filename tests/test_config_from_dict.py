"""Contracts for Config.from_dict: strict acceptance and rejection."""

from dataclasses import dataclass
from typing import ClassVar

import pytest

from slotvox.config.base import CONFIG_SCHEMA_VERSION, Config, require_int
from slotvox.errors import ConfigError


@dataclass(frozen=True)
class Demo(Config):
    kind: ClassVar[str] = "demo-from-dict"

    value: int = 2

    def validate(self) -> None:
        require_int("Demo.value", self.value, minimum=0, maximum=10)


def payload(**overrides):
    data = {"kind": "demo-from-dict", "schema_version": CONFIG_SCHEMA_VERSION, "value": 2}
    data.update(overrides)
    return data


def test_round_trip_returns_equal_config():
    original = Demo(value=7)
    assert Demo.from_dict(original.to_dict()) == original


def test_wrong_kind_rejected():
    with pytest.raises(ConfigError, match="expected config kind"):
        Demo.from_dict(payload(kind="other"))


def test_wrong_schema_version_rejected():
    with pytest.raises(ConfigError, match="schema_version"):
        Demo.from_dict(payload(schema_version=CONFIG_SCHEMA_VERSION + 1))


def test_unknown_field_rejected_by_name():
    with pytest.raises(ConfigError, match="extra"):
        Demo.from_dict(payload(extra=1))


def test_missing_field_rejected():
    data = payload()
    del data["value"]
    with pytest.raises(ConfigError, match="missing fields"):
        Demo.from_dict(data)


def test_non_dict_rejected():
    with pytest.raises(ConfigError, match="must be a dict"):
        Demo.from_dict(["kind", "demo-from-dict"])


def test_invalid_value_still_rejected_through_from_dict():
    with pytest.raises(ConfigError, match="Demo.value"):
        Demo.from_dict(payload(value=99))
