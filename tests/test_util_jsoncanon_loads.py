"""Contracts for canonical_loads: roundtrip, duplicate-key and byte handling."""

import pytest

from slotvox.errors import SchemaError
from slotvox.util.jsoncanon import canonical_dumps, canonical_loads


def test_round_trip_identity():
    obj = {"b": [1, 2.5, None, True], "a": {"嵌套": "值"}}
    assert canonical_loads(canonical_dumps(obj)) == obj


def test_duplicate_keys_rejected():
    with pytest.raises(SchemaError, match="duplicate key"):
        canonical_loads('{"a":1,"a":2}')


def test_duplicate_keys_rejected_when_nested():
    with pytest.raises(SchemaError, match="duplicate key"):
        canonical_loads('{"outer":{"a":1,"a":2}}')


def test_invalid_json_raises_schema_error():
    with pytest.raises(SchemaError, match="invalid JSON"):
        canonical_loads("{not json")


def test_bytes_input_decoded_as_utf8():
    assert canonical_loads('{"a":"值"}'.encode()) == {"a": "值"}


def test_invalid_utf8_bytes_rejected():
    with pytest.raises(SchemaError, match="utf-8"):
        canonical_loads(b'{"a":"\xff\xfe"}')
