"""Contracts for stable_hash: determinism, order-independence, sensitivity."""

import re

import pytest

from slotvox.errors import SchemaError
from slotvox.util.jsoncanon import stable_hash


def test_hash_is_lowercase_sha256_hex():
    assert re.fullmatch(r"[0-9a-f]{64}", stable_hash({"a": 1}))


def test_hash_is_deterministic_across_calls():
    obj = {"list": [1, 2, 3], "text": "语音", "nested": {"x": None}}
    assert stable_hash(obj) == stable_hash(obj)


def test_key_insertion_order_does_not_change_hash():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_hash_distinguishes_numeric_types_and_values():
    digests = {stable_hash(v) for v in ({"a": 1}, {"a": 1.0}, {"a": True}, {"a": "1"})}
    assert len(digests) == 4


def test_hash_rejects_non_canonical_values():
    with pytest.raises(SchemaError, match="non-finite"):
        stable_hash({"x": float("nan")})
    with pytest.raises(SchemaError, match="keys must be str"):
        stable_hash({1: "a"})
