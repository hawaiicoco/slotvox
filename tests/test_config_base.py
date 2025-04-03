"""Contracts for config base machinery: field validators and to_dict."""

from dataclasses import dataclass
from typing import ClassVar

import pytest

from slotvox.config.base import (
    CONFIG_SCHEMA_VERSION,
    Config,
    require_float,
    require_int,
    require_str,
)
from slotvox.errors import ConfigError


@dataclass(frozen=True)
class DemoConfig(Config):
    kind: ClassVar[str] = "demo-base"

    value: int = 2
    label: str = "ok"

    def validate(self) -> None:
        require_int("DemoConfig.value", self.value, minimum=0, maximum=10)
        require_str("DemoConfig.label", self.label, allowed=("ok", "fine"))


def test_require_int_accepts_bounds_and_rejects_outside():
    assert require_int("v", 0, minimum=0, maximum=10) == 0
    assert require_int("v", 10, minimum=0, maximum=10) == 10
    for bad in (-1, 11):
        with pytest.raises(ConfigError):
            require_int("v", bad, minimum=0, maximum=10)


def test_require_int_rejects_bools_floats_and_strings():
    for bad in (True, 2.0, "2", None):
        with pytest.raises(ConfigError, match="must be an integer"):
            require_int("v", bad)


def test_require_float_coerces_ints_and_enforces_bounds():
    assert require_float("v", 3, minimum=0.0) == 3.0
    with pytest.raises(ConfigError):
        require_float("v", -0.5, minimum=0.0)
    with pytest.raises(ConfigError, match="must be a number"):
        require_float("v", "1.0")


def test_require_float_rejects_non_finite():
    with pytest.raises(ConfigError, match="finite"):
        require_float("v", float("nan"))


def test_require_str_enforces_allowed_set():
    assert require_str("v", "ok", allowed=("ok", "fine")) == "ok"
    with pytest.raises(ConfigError, match="must be one of"):
        require_str("v", "bad", allowed=("ok", "fine"))


def test_construction_validates():
    with pytest.raises(ConfigError, match="DemoConfig.value"):
        DemoConfig(value=99)


def test_to_dict_tags_kind_and_schema_version():
    assert DemoConfig().to_dict() == {
        "kind": "demo-base",
        "schema_version": CONFIG_SCHEMA_VERSION,
        "value": 2,
        "label": "ok",
    }
