"""Identifier rules shared across schema objects.

Every named schema object (slot, intent, domain) uses kebab-case ids:
``^[a-z][a-z0-9]*(-[a-z0-9]+)*$`` with a hard length bound. One rule, one
place, validated in every constructor and parser.
"""

from __future__ import annotations

import re
from typing import Any

from slotvox.errors import ValidationError

ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
MAX_ID_LENGTH = 64


def validate_id(value: Any, field: str) -> str:
    """Validate a kebab-case identifier; return it unchanged."""
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be a string, got {type(value).__name__}")
    if not value or len(value) > MAX_ID_LENGTH:
        raise ValidationError(
            f"{field} must be 1..{MAX_ID_LENGTH} characters, got length {len(value)}"
        )
    if not ID_PATTERN.fullmatch(value):
        raise ValidationError(
            f"{field} must be kebab-case (^[a-z][a-z0-9]*(-[a-z0-9]+)*$), got {value!r}"
        )
    return value
