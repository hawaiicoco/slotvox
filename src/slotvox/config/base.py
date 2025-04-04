"""Shared machinery for slotvox configuration records.

Configs are frozen dataclasses that validate on construction and serialize
to canonical dicts tagged with an explicit ``kind`` and ``schema_version``.
Invalid configs raise :class:`slotvox.errors.ConfigError` naming the
offending field, so configuration mistakes surface at construction time
rather than deep inside a pipeline.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar, Self

from slotvox.errors import ConfigError
from slotvox.util.jsoncanon import canonical_dumps, canonical_loads

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Reconstruct a config, strictly rejecting wrong kind/version/fields.

        Every declared field must be present (``to_dict`` always emits all of
        them), and no extra fields are accepted; this keeps serialized
        configs unambiguous in both directions.
        """
        if not isinstance(data, dict):
            raise ConfigError(f"config payload must be a dict, got {type(data).__name__}")
        kind = data.get("kind")
        if kind != cls.kind:
            raise ConfigError(f"expected config kind {cls.kind!r}, got {kind!r}")
        version = data.get("schema_version")
        if version != CONFIG_SCHEMA_VERSION:
            raise ConfigError(
                f"config kind {cls.kind!r} expects schema_version "
                f"{CONFIG_SCHEMA_VERSION}, got {version!r}"
            )
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known - {"kind", "schema_version"}
        if unknown:
            raise ConfigError(f"config kind {cls.kind!r} got unknown fields: {sorted(unknown)}")
        missing = known - set(data)
        if missing:
            raise ConfigError(f"config kind {cls.kind!r} is missing fields: {sorted(missing)}")
        return cls(**{name: data[name] for name in known})


_REGISTRY: dict[str, type[Config]] = {}


def register_config(cls: type[Config]) -> type[Config]:
    """Class decorator making ``cls`` reachable via :func:`config_from_dict`."""
    if not isinstance(cls.kind, str) or not cls.kind or cls.kind == "config":
        raise ConfigError(f"{cls.__name__} must define a concrete kind, got {cls.kind!r}")
    previous = _REGISTRY.get(cls.kind)
    if previous is not None and previous is not cls:
        raise ConfigError(f"duplicate config kind: {cls.kind!r}")
    _REGISTRY[cls.kind] = cls
    return cls


def registered_kinds() -> tuple[str, ...]:
    """All config kinds reachable from :func:`config_from_dict`, sorted."""
    return tuple(sorted(_REGISTRY))


def config_from_dict(data: dict[str, Any]) -> Config:
    """Dispatch to the registered config class named by ``data["kind"]``."""
    if not isinstance(data, dict):
        raise ConfigError(f"config payload must be a dict, got {type(data).__name__}")
    kind = data.get("kind")
    cls = _REGISTRY.get(kind) if isinstance(kind, str) else None
    if cls is None:
        raise ConfigError(f"unknown config kind: {kind!r} (registered: {list(registered_kinds())})")
    return cls.from_dict(data)


def config_to_json(config: Config) -> str:
    """Canonical JSON text for ``config``."""
    return canonical_dumps(config.to_dict())


def config_from_json(text: str | bytes) -> Config:
    """Parse canonical JSON text into the registered config it names."""
    return config_from_dict(canonical_loads(text))
