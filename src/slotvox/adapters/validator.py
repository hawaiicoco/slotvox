"""A miniature JSON-schema validator for the adapter protocol.

Supports exactly the keyword subset the protocol schemas use: ``type``,
``required``, ``properties``, ``additionalProperties``, ``items``,
``enum``, ``minimum``, ``maximum``, ``minLength``, ``minItems``,
``maxItems``, ``pattern``. Unsupported keywords are rejected outright
rather than silently ignored, and errors carry a JSON-pointer-style path
so a failure points at the offending field. This is a protocol tool, not
a general schema engine.
"""

from __future__ import annotations

import re
from typing import Any

from slotvox.errors import AdapterError

SUPPORTED_KEYWORDS = frozenset(
    {
        "type",
        "required",
        "properties",
        "additionalProperties",
        "items",
        "enum",
        "minimum",
        "maximum",
        "minLength",
        "minItems",
        "maxItems",
        "pattern",
    }
)

_TYPE_CHECKS = {
    "object": lambda value: isinstance(value, dict),
    "array": lambda value: isinstance(value, list),
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "null": lambda value: value is None,
}


def validate(instance: Any, schema: Any, path: str = "$") -> None:
    """Validate ``instance`` against the mini ``schema``; raise AdapterError."""
    if not isinstance(schema, dict):
        raise AdapterError(f"schema at {path} must be a dict, got {type(schema).__name__}")
    unsupported = sorted(set(schema) - SUPPORTED_KEYWORDS)
    if unsupported:
        raise AdapterError(f"unsupported schema keywords at {path}: {unsupported}")
    expected = schema.get("type")
    if expected is not None:
        if expected not in _TYPE_CHECKS:
            raise AdapterError(f"unknown type {expected!r} in schema at {path}")
        if not _TYPE_CHECKS[expected](instance):
            raise AdapterError(f"{path}: expected type {expected!r}, got {type(instance).__name__}")
    if "enum" in schema and not any(
        instance is option or instance == option for option in schema["enum"]
    ):
        raise AdapterError(f"{path}: {instance!r} is not one of {schema['enum']}")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            raise AdapterError(f"{path}: {instance!r} is below minimum {schema['minimum']!r}")
        if "maximum" in schema and instance > schema["maximum"]:
            raise AdapterError(f"{path}: {instance!r} is above maximum {schema['maximum']!r}")
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            raise AdapterError(f"{path}: string is shorter than {schema['minLength']}")
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            raise AdapterError(f"{path}: string does not match pattern {schema['pattern']!r}")
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            raise AdapterError(f"{path}: array has fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            raise AdapterError(f"{path}: array has more than {schema['maxItems']} items")
        if "items" in schema:
            for index, element in enumerate(instance):
                validate(element, schema["items"], f"{path}[{index}]")
    if isinstance(instance, dict):
        required = schema.get("required", [])
        missing = [key for key in required if key not in instance]
        if missing:
            raise AdapterError(f"{path}: missing required keys: {missing}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = sorted(set(instance) - set(properties))
            if extra:
                raise AdapterError(f"{path}: additional properties are not allowed: {extra}")
        for key, subschema in properties.items():
            if key in instance:
                validate(instance[key], subschema, f"{path}.{key}")
