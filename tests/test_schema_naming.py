"""Identifier rule contracts: acceptance and rejection."""

import pytest

from slotvox.errors import ValidationError
from slotvox.schema.naming import MAX_ID_LENGTH, validate_id


@pytest.mark.parametrize(
    "value", ["a", "weather", "music-control", "play-music", "slot-2", "x-1-y", "a" * MAX_ID_LENGTH]
)
def test_valid_ids_accepted(value):
    assert validate_id(value, "id") == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        "Weather",
        "we_ather",
        "-lead",
        "trail-",
        "double--dash",
        "sp ace",
        "weathe.r",
        "1digit",
        "a" * (MAX_ID_LENGTH + 1),
        42,
        None,
    ],
)
def test_invalid_ids_rejected(value):
    with pytest.raises(ValidationError):
        validate_id(value, "id")


def test_field_name_appears_in_message():
    with pytest.raises(ValidationError, match="IntentDefinition.name"):
        validate_id("Bad", "IntentDefinition.name")
