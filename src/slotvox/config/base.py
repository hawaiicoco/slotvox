"""Shared machinery for slotvox configuration records.

Configs are frozen dataclasses that validate on construction and serialize
to canonical dicts tagged with an explicit ``kind`` and ``schema_version``.
Invalid configs raise :class:`slotvox.errors.ConfigError` naming the
offending field, so configuration mistakes surface at construction time
rather than deep inside a pipeline.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from slotvox.errors import ConfigError

CONFIG_SCHEMA_VERSION = 1


def require_int(
    name: str,
    value: Any,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Validate that ``value`` is a plain int within bounds; return it."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be <= {maximum}, got {value}")
    return value


def require_float(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Validate that ``value`` is a finite number within bounds; return it."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{name} must be finite, got {result!r}")
    if minimum is not None and result < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise ConfigError(f"{name} must be <= {maximum}, got {result}")
    return result


def require_str(name: str, value: Any, *, allowed: tuple[str, ...] | None = None) -> str:
    """Validate that ``value`` is a string, optionally from ``allowed``."""
    if not isinstance(value, str):
        raise ConfigError(f"{name} must be a string, got {value!r}")
    if allowed is not None and value not in allowed:
        raise ConfigError(f"{name} must be one of {sorted(allowed)}, got {value!r}")
    return value


@dataclass(frozen=True)
class Config:
    """Base class for validated, versioned configuration records.

    Subclasses declare a concrete ``kind`` ClassVar, give every field a
    default, and override :meth:`validate` to enforce their contracts.
    """

    kind: ClassVar[str] = "config"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Enforce field contracts; the base record accepts anything."""

    def to_dict(self) -> dict[str, Any]:
        """Canonical dict form: fields plus ``kind`` and ``schema_version``."""
        data: dict[str, Any] = asdict(self)
        data["kind"] = self.kind
        data["schema_version"] = CONFIG_SCHEMA_VERSION
        return data
