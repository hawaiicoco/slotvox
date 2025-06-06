"""Slot type specifications with strict value constraints.

A slot type names what kind of value fills a slot — an enum choice, a
bounded number, a clock time, or free text — plus the constraint payload
for that kind. Constraint keys are exhaustive per type: unknown keys are
rejected, required keys must be present, and every spec normalizes its
constraint at construction so serialization round-trips exactly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from slotvox.errors import ValidationError
from slotvox.schema.naming import validate_id

SLOT_VALUE_TYPES = ("enum", "number", "time", "text")

TIME_PATTERN = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

TEXT_DEFAULT_MIN = 1
TEXT_DEFAULT_MAX = 512


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"{field} must be a finite number, got {value!r}")
    return float(value)


def _reject_unknown_keys(
    value_type: str, constraint: Any, allowed: tuple[str, ...], name: str
) -> None:
    if not isinstance(constraint, dict):
        raise ValidationError(
            f"slot {name!r} constraint must be a dict, got {type(constraint).__name__}"
        )
    unknown = sorted(set(constraint) - set(allowed))
    if unknown:
        raise ValidationError(
            f"slot {name!r} ({value_type}) constraint got unknown keys: {unknown}"
        )


def _normalize_constraint(value_type: str, constraint: Any, name: str) -> dict[str, Any]:
    """Validate and normalize the constraint payload for ``value_type``."""
    if value_type == "enum":
        _reject_unknown_keys(value_type, constraint, ("choices",), name)
        if "choices" not in constraint:
            raise ValidationError(f"enum slot {name!r} requires 'choices'")
        choices = constraint["choices"]
        if isinstance(choices, (str, bytes)) or not isinstance(choices, (list, tuple)):
            raise ValidationError(f"enum slot {name!r} choices must be a list of strings")
        if not choices:
            raise ValidationError(f"enum slot {name!r} choices must be non-empty")
        for choice in choices:
            if not isinstance(choice, str) or not choice:
                raise ValidationError(
                    f"enum slot {name!r} choices must be non-empty strings, got {choice!r}"
                )
        if len(set(choices)) != len(choices):
            raise ValidationError(f"enum slot {name!r} choices contain duplicates")
        return {"choices": tuple(choices)}
    if value_type == "number":
        _reject_unknown_keys(value_type, constraint, ("minimum", "maximum", "unit"), name)
        normalized: dict[str, Any] = {}
        for key in ("minimum", "maximum"):
            if key in constraint:
                normalized[key] = _finite_number(constraint[key], f"number slot {name!r} {key}")
        if (
            "minimum" in normalized
            and "maximum" in normalized
            and normalized["minimum"] > normalized["maximum"]
        ):
            raise ValidationError(f"number slot {name!r} requires minimum <= maximum")
        if "unit" in constraint:
            unit = constraint["unit"]
            if not isinstance(unit, str) or not unit:
                raise ValidationError(f"number slot {name!r} unit must be a non-empty string")
            normalized["unit"] = unit
        return normalized
    if value_type == "time":
        _reject_unknown_keys(value_type, constraint, (), name)
        return {}
    _reject_unknown_keys(value_type, constraint, ("min_length", "max_length"), name)
    lengths: dict[str, int] = {}
    for key, default in (("min_length", TEXT_DEFAULT_MIN), ("max_length", TEXT_DEFAULT_MAX)):
        value = constraint.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1000:
            raise ValidationError(
                f"text slot {name!r} {key} must be an int within [1, 1000], got {value!r}"
            )
        lengths[key] = value
    if lengths["min_length"] > lengths["max_length"]:
        raise ValidationError(f"text slot {name!r} requires min_length <= max_length")
    return lengths


@dataclass(frozen=True)
class SlotTypeSpec:
    """One slot type of a domain vocabulary."""

    name: str
    value_type: str
    constraint: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        validate_id(self.name, "SlotTypeSpec.name")
        if self.value_type not in SLOT_VALUE_TYPES:
            raise ValidationError(
                f"unknown slot value type {self.value_type!r}; expected one of {SLOT_VALUE_TYPES}"
            )
        if not isinstance(self.description, str):
            raise ValidationError("SlotTypeSpec.description must be a string")
        object.__setattr__(
            self,
            "constraint",
            _normalize_constraint(self.value_type, self.constraint, self.name),
        )

    def check_value(self, value: Any) -> None:
        """Validate a concrete slot value against this spec.

        Accepts exactly the values the constraint describes and rejects
        everything else with a message naming the slot — validation in both
        directions is part of the contract.
        """
        if self.value_type == "enum":
            if not isinstance(value, str) or value not in self.constraint["choices"]:
                raise ValidationError(
                    f"slot {self.name!r} value {value!r} is not one of the allowed choices"
                )
        elif self.value_type == "number":
            number = _finite_number(value, f"slot {self.name!r} value")
            minimum = self.constraint.get("minimum")
            maximum = self.constraint.get("maximum")
            if minimum is not None and number < minimum:
                raise ValidationError(
                    f"slot {self.name!r} value {number} is below minimum {minimum}"
                )
            if maximum is not None and number > maximum:
                raise ValidationError(
                    f"slot {self.name!r} value {number} exceeds maximum {maximum}"
                )
        elif self.value_type == "time":
            if not isinstance(value, str) or not TIME_PATTERN.fullmatch(value):
                raise ValidationError(
                    f"slot {self.name!r} requires a 24h HH:MM time string, got {value!r}"
                )
        else:
            if not isinstance(value, str):
                raise ValidationError(
                    f"slot {self.name!r} requires a string value, got {type(value).__name__}"
                )
            min_length = self.constraint["min_length"]
            max_length = self.constraint["max_length"]
            if not min_length <= len(value) <= max_length:
                raise ValidationError(
                    f"slot {self.name!r} value length {len(value)} is outside "
                    f"[{min_length}, {max_length}]"
                )
