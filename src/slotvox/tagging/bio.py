"""BIO/BIOES tag algebra with validity checking and repair policies.

Conventions: ``O`` marks tokens outside any slot span; ``B-<slot>`` starts
a span; ``I-<slot>`` continues the open span of the same slot. BIOES adds
``E-<slot>`` (end of a multi-token span) and ``S-<slot>`` (single-token
span). Sequences are validated structurally, repaired under explicit
policies, and converted losslessly between BIO and BIOES.

This module is pure label algebra: no audio, no text, no model. It is the
authority the data factory and the evaluation metrics defer to.
"""

from __future__ import annotations

from slotvox.errors import TaggingError, ValidationError
from slotvox.schema.naming import validate_id

BIO_PREFIXES = ("B", "I")
BIOES_PREFIXES = ("B", "I", "E", "S")
OUTSIDE = "O"
REPAIR_POLICIES = ("strict", "drop", "promote")


def _check_slot_id(slot: str, field: str) -> str:
    """Validate a slot id, surfacing failures as TaggingError."""
    try:
        return validate_id(slot, field)
    except ValidationError as exc:
        raise TaggingError(str(exc)) from exc


def parse_tag(tag: str) -> tuple[str, str | None]:
    """Split a tag into ``(prefix, slot)``; ``("O", None)`` for outside.

    Raises TaggingError for anything malformed — parsing is the validation
    gate every other function goes through.
    """
    if not isinstance(tag, str):
        raise TaggingError(f"tag must be a string, got {type(tag).__name__}")
    if tag == OUTSIDE:
        return OUTSIDE, None
    if "-" not in tag:
        raise TaggingError(f"tag {tag!r} must be 'O' or '<prefix>-<slot>'")
    prefix, slot = tag.split("-", 1)
    if prefix not in BIOES_PREFIXES:
        raise TaggingError(
            f"tag {tag!r} has unknown prefix {prefix!r}; expected one of {BIOES_PREFIXES} or 'O'"
        )
    _check_slot_id(slot, f"tag slot name in {tag!r}")
    return prefix, slot


def format_tag(prefix: str, slot: str | None) -> str:
    """Inverse of :func:`parse_tag` (strict in both arguments)."""
    if prefix == OUTSIDE:
        if slot is not None:
            raise TaggingError(f"'O' tag cannot carry a slot, got {slot!r}")
        return OUTSIDE
    if prefix not in BIOES_PREFIXES:
        raise TaggingError(f"unknown tag prefix {prefix!r}")
    if not isinstance(slot, str) or not slot:
        raise TaggingError(f"prefix {prefix!r} requires a slot name, got {slot!r}")
    _check_slot_id(slot, "format_tag slot")
    return f"{prefix}-{slot}"


def is_valid_tag(tag: str, *, bioes: bool = False) -> bool:
    """True iff ``tag`` parses; ``bioes=False`` rejects E-/S- prefixes."""
    try:
        prefix, _ = parse_tag(tag)
    except TaggingError:
        return False
    if not bioes and prefix in ("E", "S"):
        return False
    return True
