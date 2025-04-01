"""Canonical JSON serialization and stable hashing.

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

import hashlib
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


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    for key, _ in pairs:
        if key in seen:
            raise SchemaError(f"duplicate key in canonical JSON object: {key!r}")
        seen.add(key)
    return dict(pairs)


def canonical_loads(text: str | bytes) -> Any:
    """Parse JSON text, rejecting duplicate object keys and invalid bytes.

    Duplicate keys make artifacts ambiguous, so canonical form forbids them
    in both directions (serialization cannot produce them; parsing rejects
    hand-edited or hostile input).
    """
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SchemaError(f"canonical JSON must be valid utf-8: {exc}") from exc
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicates)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"invalid JSON: {exc}") from exc


def stable_hash(obj: Any) -> str:
    """SHA-256 hex digest of the canonical JSON form of ``obj``.

    Logically equal artifacts hash identically across processes and runs;
    any change in structure, value, or numeric type changes the digest.
    """
    return hashlib.sha256(canonical_dumps(obj).encode("utf-8")).hexdigest()
