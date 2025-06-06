"""SlotTypeSpec.check_value contracts, both directions, per type."""

import pytest

from slotvox.errors import ValidationError
from slotvox.schema.slots import SlotTypeSpec


def test_enum_values():
    spec = SlotTypeSpec("condition", "enum", {"choices": ["sunny", "rainy"]})
    spec.check_value("sunny")
    spec.check_value("rainy")
    for bad in ("cloudy", "", 42, None, ["sunny"]):
        with pytest.raises(ValidationError, match="not one of the allowed choices"):
            spec.check_value(bad)


def test_number_values_respect_bounds():
    spec = SlotTypeSpec("temperature", "number", {"minimum": -40.0, "maximum": 60.0})
    for good in (-40.0, 0, 23.5, 60.0):
        spec.check_value(good)
    for bad in (-40.5, 60.5, "23", True, float("nan")):
        with pytest.raises(ValidationError):
            spec.check_value(bad)


def test_unbounded_number_accepts_any_finite():
    spec = SlotTypeSpec("distance", "number")
    spec.check_value(-1e9)
    spec.check_value(1e9)
    with pytest.raises(ValidationError, match="below minimum"):
        SlotTypeSpec("t", "number", {"minimum": 0.0}).check_value(-0.1)
    with pytest.raises(ValidationError, match="exceeds maximum"):
        SlotTypeSpec("t", "number", {"maximum": 0.0}).check_value(0.1)


def test_time_values():
    spec = SlotTypeSpec("depart", "time")
    for good in ("00:00", "07:30", "23:59"):
        spec.check_value(good)
    for bad in ("24:00", "7:30", "07:60", "0730", "", 730, "07:30:00"):
        with pytest.raises(ValidationError, match="HH:MM"):
            spec.check_value(bad)


def test_text_values_respect_lengths():
    default = SlotTypeSpec("note", "text")
    default.check_value("hello")
    with pytest.raises(ValidationError, match="length"):
        default.check_value("")
    with pytest.raises(ValidationError, match="length"):
        default.check_value("x" * 513)
    custom = SlotTypeSpec("code", "text", {"min_length": 2, "max_length": 4})
    custom.check_value("ab")
    custom.check_value("abcd")
    for bad in ("a", "abcde", 12, None):
        with pytest.raises(ValidationError):
            custom.check_value(bad)
