"""Lexicon machinery contracts: lookup, validation, registration."""

import pytest

from slotvox.data.lexicon import (
    LexiconEntry,
    register_lexicon,
    registered_lexicons,
    slot_entries,
    validate_lexicon,
)
from slotvox.errors import ValidationError
from slotvox.schema.builtins import builtin_domain


def test_weather_zh_lexicon_is_registered():
    assert ("weather", "zh") in registered_lexicons()
    entries = slot_entries("weather", "zh", "city")
    assert entries[0] == LexiconEntry("北京", "北京")


def test_strict_lookup_rejections():
    for args in [
        ("nope", "zh", "city"),
        ("weather", "fr", "city"),
        ("weather", "zh", "nope"),
    ]:
        with pytest.raises(ValidationError):
            slot_entries(*args)


def test_entry_surface_rules():
    with pytest.raises(ValidationError):
        LexiconEntry("", "x")
    with pytest.raises(ValidationError):
        LexiconEntry(" pad ", "x")
    with pytest.raises(ValidationError):
        LexiconEntry(None, "x")


def test_validate_lexicon_rejects_bad_content():
    domain = builtin_domain("weather")
    with pytest.raises(ValidationError):
        validate_lexicon(domain, {"city": (LexiconEntry("成都", "成都"),)})
    with pytest.raises(ValidationError):
        validate_lexicon(domain, {"nope": (LexiconEntry("x", "x"),)})
    with pytest.raises(ValidationError):
        validate_lexicon(domain, {"city": ()})
    with pytest.raises(ValidationError):
        validate_lexicon(domain, {})
    with pytest.raises(ValidationError):
        validate_lexicon("weather", {"city": (LexiconEntry("北京", "北京"),)})


def test_duplicate_registration_rejected():
    with pytest.raises(ValidationError, match="already registered"):
        register_lexicon("weather", "zh", {"city": (LexiconEntry("深圳", "深圳"),)})
    # the failed attempt must not have mutated anything
    assert slot_entries("weather", "zh", "city")[0].surface == "北京"
