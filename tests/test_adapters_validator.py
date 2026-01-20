"""Mini JSON-schema validator: every keyword, both directions."""

import pytest

from slotvox.adapters.validator import validate
from slotvox.errors import AdapterError


def test_object_rules():
    schema = {
        "type": "object",
        "required": ["a"],
        "additionalProperties": False,
        "properties": {"a": {"type": "string", "minLength": 2}},
    }
    validate({"a": "xy"}, schema)
    with pytest.raises(AdapterError, match=r"\$: expected type"):
        validate([], schema)
    with pytest.raises(AdapterError, match="missing required"):
        validate({}, schema)
    with pytest.raises(AdapterError, match="additional"):
        validate({"a": "xy", "b": 1}, schema)
    with pytest.raises(AdapterError, match=r"\$\.a"):
        validate({"a": 1}, schema)
    with pytest.raises(AdapterError, match="shorter"):
        validate({"a": "x"}, schema)


def test_scalar_rules():
    validate(3, {"type": "integer", "minimum": 1, "maximum": 5})
    with pytest.raises(AdapterError, match="minimum"):
        validate(0, {"type": "integer", "minimum": 1})
    with pytest.raises(AdapterError, match="maximum"):
        validate(6, {"type": "integer", "maximum": 5})
    with pytest.raises(AdapterError, match="expected type"):
        validate(True, {"type": "integer"})  # bools are not integers here
    validate(1, {"type": "number"})
    validate("abc", {"enum": ["abc", "def"]})
    with pytest.raises(AdapterError, match="not one of"):
        validate("xyz", {"enum": ["abc", "def"]})
    validate("a1b", {"type": "string", "pattern": r"^[a-z]\d"})
    with pytest.raises(AdapterError, match="pattern"):
        validate("1ab", {"type": "string", "pattern": r"^[a-z]\d"})


def test_array_rules_and_paths():
    schema = {"type": "array", "minItems": 1, "maxItems": 2, "items": {"type": "integer"}}
    validate([1, 2], schema)
    with pytest.raises(AdapterError, match="minItems|fewer"):
        validate([], schema)
    with pytest.raises(AdapterError, match="more than"):
        validate([1, 2, 3], schema)
    with pytest.raises(AdapterError, match=r"\$\[1\]"):
        validate([1, "x"], schema)


def test_unsupported_keyword_and_bad_schema():
    with pytest.raises(AdapterError, match="unsupported"):
        validate({}, {"type": "object", "uniqueItems": True})
    with pytest.raises(AdapterError, match="must be a dict"):
        validate({}, "schema")
    with pytest.raises(AdapterError, match="unknown type"):
        validate(1, {"type": "matrix"})
