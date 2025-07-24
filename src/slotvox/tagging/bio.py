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


def _sequence(value, name: str = "tags") -> tuple:
    """Coerce a list/tuple of items, rejecting strings and other types."""
    if isinstance(value, (str, bytes)):
        raise TaggingError(f"{name} must be a sequence of strings, got {type(value).__name__}")
    if not isinstance(value, (list, tuple)):
        raise TaggingError(f"{name} must be a list/tuple, got {type(value).__name__}")
    return tuple(value)


def validate_sequence(tags, *, bioes: bool = False) -> None:
    """Raise TaggingError unless ``tags`` is structurally valid.

    BIO rules: an ``I-x`` must continue an open ``x`` span (previous tag
    ``B-x`` or ``I-x``). BIOES additionally requires spans opened with
    ``B-x`` to close with ``E-x`` before anything else starts, ``S-x`` to
    stand alone, and no unclosed span at the end.
    """
    items = _sequence(tags)
    open_slot: str | None = None
    for index, tag in enumerate(items):
        prefix, slot = parse_tag(tag)
        if not bioes and prefix in ("E", "S"):
            raise TaggingError(f"tag {index} ({tag!r}) is BIOES-only; pass bioes=True")
        if prefix == OUTSIDE:
            open_slot = None
        elif prefix == "B":
            if bioes and open_slot is not None:
                raise TaggingError(
                    f"tag {index} ({tag!r}) starts a span before closing {open_slot!r}"
                )
            open_slot = slot
        elif prefix == "S":
            if open_slot is not None:
                raise TaggingError(f"tag {index} ({tag!r}) appears inside an open span")
        elif prefix == "I":
            if slot != open_slot:
                raise TaggingError(
                    f"tag {index} ({tag!r}) does not continue an open span (open: {open_slot!r})"
                )
        else:  # "E"
            if slot != open_slot:
                raise TaggingError(
                    f"tag {index} ({tag!r}) closes span {open_slot!r} it never opened"
                )
            open_slot = None
    if bioes and open_slot is not None:
        raise TaggingError(f"sequence ends with an unclosed span {open_slot!r} (missing E/S)")


def is_valid_sequence(tags, *, bioes: bool = False) -> bool:
    """Non-raising companion of :func:`validate_sequence`."""
    try:
        validate_sequence(tags, bioes=bioes)
        return True
    except TaggingError:
        return False
