"""Canonical JSON serialization.

Canonical form (slotvox canonical-json v1):

- object keys must be ``str`` and are sorted lexicographically,
- separators are ``(",", ":")`` (no insignificant whitespace),
- ``ensure_ascii=False`` so text is emitted as UTF-8,
- non-finite floats are rejected (they are not valid JSON),
- only JSON-native values are allowed: ``None``, ``bool``, ``int``,
  ``float``, ``str``, lists/tuples (emitted as arrays), and dicts.

Canonicalization makes serialized artifacts byte-identical for logically
equal inputs, which is what provenance hashes and golden tests rely on.
"""

from __future__ import annotations

import json
import math
from typing import Any

from slotvox.errors import SchemaError

_SCALARS = (str, int, float, bool, type(None))


def _validate(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaError(
                    f"canonical JSON object keys must be str, got {type(key).__name__}"
                )
            _validate(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate(item)
    elif isinstance(value, _SCALARS):
        if isinstance(value, float) and not math.isfinite(value):
            raise SchemaError(f"canonical JSON rejects non-finite floats, got {value!r}")
    else:
        raise SchemaError(f"object of type {type(value).__name__} is not allowed in canonical JSON")


def canonical_dumps(obj: Any) -> str:
    """Serialize ``obj`` to its canonical JSON text form."""
    _validate(obj)
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
