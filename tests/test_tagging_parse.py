"""parse_tag/format_tag/is_valid_tag contracts, both directions."""

import pytest

from slotvox.errors import TaggingError
from slotvox.tagging.bio import format_tag, is_valid_tag, parse_tag


def test_parse_outside():
    assert parse_tag("O") == ("O", None)


@pytest.mark.parametrize("prefix", ["B", "I", "E", "S"])
def test_parse_prefixed_tags(prefix):
    assert parse_tag(f"{prefix}-city") == (prefix, "city")


def test_parse_accepts_kebab_slots():
    assert parse_tag("B-volume-level") == ("B", "volume-level")


@pytest.mark.parametrize(
    "tag", ["", "city", "X-city", "B-", "b-city", "B-Bad", "B--x", "B-sp ace", None, 42]
)
def test_parse_rejects_malformed(tag):
    with pytest.raises(TaggingError):
        parse_tag(tag)


def test_format_is_inverse_of_parse():
    assert format_tag("O", None) == "O"
    assert format_tag("B", "city") == "B-city"
    for prefix in ("B", "I", "E", "S"):
        assert parse_tag(format_tag(prefix, "slot-x")) == (prefix, "slot-x")


@pytest.mark.parametrize(
    ("prefix", "slot"), [("O", "x"), ("B", None), ("B", ""), ("Q", "x"), ("B", "Bad")]
)
def test_format_rejects_bad_combinations(prefix, slot):
    with pytest.raises(TaggingError):
        format_tag(prefix, slot)


def test_is_valid_tag_modes():
    assert is_valid_tag("O")
    assert is_valid_tag("B-x")
    assert not is_valid_tag("S-x")
    assert is_valid_tag("S-x", bioes=True)
    assert is_valid_tag("E-x", bioes=True)
    assert not is_valid_tag("nope")
