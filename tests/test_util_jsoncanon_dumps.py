"""Contracts for canonical_dumps: ordering, separators, unicode, rejection."""

import pytest

from slotvox.errors import SchemaError
from slotvox.util.jsoncanon import canonical_dumps


def test_keys_sorted_and_separators_compact():
    text = canonical_dumps({"b": 1, "a": [2, {"d": 4, "c": 3}]})
    assert text == '{"a":[2,{"c":3,"d":4}],"b":1}'


def test_unicode_preserved_as_utf8():
    assert canonical_dumps({"文本": "语音"}) == '{"文本":"语音"}'


def test_tuples_serialize_as_arrays():
    assert canonical_dumps((1, "two", None, True)) == '[1,"two",null,true]'


def test_non_finite_floats_rejected():
    with pytest.raises(SchemaError, match="non-finite"):
        canonical_dumps({"x": float("nan")})
    with pytest.raises(SchemaError, match="non-finite"):
        canonical_dumps({"x": float("inf")})


def test_non_string_keys_rejected():
    with pytest.raises(SchemaError, match="keys must be str"):
        canonical_dumps({1: "a"})


def test_foreign_types_rejected():
    with pytest.raises(SchemaError, match="not allowed"):
        canonical_dumps({"x": object()})


def test_bool_and_int_stay_distinct():
    assert canonical_dumps({"a": True}) == '{"a":true}'
    assert canonical_dumps({"a": 1}) == '{"a":1}'
