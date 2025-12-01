"""Strict template rendering for instruction turns.

Templates carry ``{variable}`` placeholders (lowercase ids). Rendering
validates BOTH directions: every placeholder must be supplied and every
supplied variable must be used — no silent drops, no leftovers. Values
must be non-blank strings without braces, so a slot surface form can
never smuggle a placeholder into rendered text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from slotvox.errors import ValidationError
from slotvox.instructions.schema import TURN_ROLES, Turn
from slotvox.schema.naming import validate_id

PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*)\}")


def template_variables(text: str) -> tuple[str, ...]:
    """Sorted unique ``{variable}`` names in ``text``; rejects stray braces."""
    if not isinstance(text, str) or not text.strip():
        raise ValidationError("template text must be a non-blank string")
    stripped = PLACEHOLDER.sub("", text)
    if "{" in stripped or "}" in stripped:
        raise ValidationError(f"template text has malformed braces: {text!r}")
    return tuple(sorted({match.group(1) for match in PLACEHOLDER.finditer(text)}))


@dataclass(frozen=True)
class TurnTemplate:
    """A role-tagged text template with ``{variable}`` placeholders."""

    template_id: str
    role: str
    text: str

    def __post_init__(self) -> None:
        validate_id(self.template_id, "TurnTemplate.template_id")
        if self.role not in TURN_ROLES:
            raise ValidationError(f"role must be one of {TURN_ROLES}, got {self.role!r}")
        template_variables(self.text)  # strict brace syntax

    @property
    def variables(self) -> tuple[str, ...]:
        """Sorted unique placeholder names."""
        return template_variables(self.text)

    def render(self, variables: Mapping[str, str]) -> Turn:
        """Fill every placeholder; strict in both directions."""
        if not isinstance(variables, Mapping):
            raise ValidationError(f"variables must be a mapping, got {type(variables).__name__}")
        expected = set(self.variables)
        supplied = set(variables)
        missing = sorted(expected - supplied)
        unknown = sorted(supplied - expected)
        if missing:
            raise ValidationError(f"template {self.template_id!r} is missing variables: {missing}")
        if unknown:
            raise ValidationError(f"template {self.template_id!r} got unknown variables: {unknown}")
        for name, value in variables.items():
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"variable {name!r} must be a non-blank string")
            if "{" in value or "}" in value:
                raise ValidationError(f"variable {name!r} must not contain braces")
        text = PLACEHOLDER.sub(lambda match: str(variables[match.group(1)]), self.text)
        return Turn(role=self.role, text=text)
