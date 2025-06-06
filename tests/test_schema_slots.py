"""SlotTypeSpec construction contracts."""

import pytest

from slotvox.errors import ValidationError
from slotvox.schema.slots import SLOT_VALUE_TYPES, SlotTypeSpec


def test_enum_spec_normalizes_choices_to_tuple():
    spec = SlotTypeSpec("city", "enum", {"choices": ["上海", "北京"]})
    assert spec.constraint["choices"] == ("上海", "北京")


def test_text_spec_applies_default_lengths():
    spec = SlotTypeSpec("note", "text")
    assert spec.constraint == {"min_length": 1, "max_length": 512}


def test_number_and_time_specs_construct():
    number = SlotTypeSpec("temperature", "number", {"minimum": -40.0, "maximum": 60, "unit": "°C"})
    assert number.constraint == {"minimum": -40.0, "maximum": 60.0, "unit": "°C"}
    assert SlotTypeSpec("depart", "time").constraint == {}


def test_unknown_value_type_rejected():
    with pytest.raises(ValidationError, match="unknown slot value type"):
        SlotTypeSpec("x", "boolean")
    assert set(SLOT_VALUE_TYPES) == {"enum", "number", "time", "text"}


@pytest.mark.parametrize("name", ["City", "", "has space"])
def test_invalid_names_rejected(name):
    with pytest.raises(ValidationError):
        SlotTypeSpec(name, "time")


@pytest.mark.parametrize(
    "constraint",
    [
        {},
        {"choices": []},
        {"choices": ["a", "a"]},
        {"choices": "ab"},
        {"choices": [1, 2]},
        {"choices": ["a"], "extra": 1},
    ],
)
def test_invalid_enum_constraints_rejected(constraint):
    with pytest.raises(ValidationError):
        SlotTypeSpec("city", "enum", constraint)


@pytest.mark.parametrize(
    "constraint",
    [
        {"minimum": 10.0, "maximum": 1.0},
        {"minimum": float("nan")},
        {"minimum": True},
        {"unit": ""},
        {"scale": 2},
    ],
)
def test_invalid_number_constraints_rejected(constraint):
    with pytest.raises(ValidationError):
        SlotTypeSpec("temperature", "number", constraint)


@pytest.mark.parametrize(
    "constraint",
    [
        {"min_length": 5, "max_length": 2},
        {"min_length": 0},
        {"max_length": 1001},
        {"min_length": 2.0},
        {"pattern": ".*"},
    ],
)
def test_invalid_text_constraints_rejected(constraint):
    with pytest.raises(ValidationError):
        SlotTypeSpec("note", "text", constraint)


def test_time_constraint_rejects_any_keys():
    with pytest.raises(ValidationError, match="unknown keys"):
        SlotTypeSpec("depart", "time", {"granularity": "minute"})
    with pytest.raises(ValidationError):
        SlotTypeSpec("depart", "time", "none")


def test_description_must_be_string():
    with pytest.raises(ValidationError, match="description"):
        SlotTypeSpec("city", "time", description=42)
