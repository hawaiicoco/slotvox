"""Dialogue domains: versioned bindings of slot vocabularies and intents.

A domain is the unit of task definition: it owns the slot vocabulary and
the intents that may reference it. Construction is strict in both
directions — duplicate names and unknown slot references are rejected —
and the whole definition round-trips through canonical JSON under an
explicit schema id and version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from slotvox.errors import SchemaError, ValidationError
from slotvox.schema.intents import IntentDefinition
from slotvox.schema.naming import validate_id
from slotvox.schema.slots import SlotTypeSpec
from slotvox.util.jsoncanon import stable_hash

DOMAIN_SCHEMA_ID = "slotvox.domain"
DOMAIN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DomainSpec:
    """A complete dialogue-domain definition."""

    domain: str
    slots: tuple[SlotTypeSpec, ...]
    intents: tuple[IntentDefinition, ...]
    description: str = ""
    schema_version: int = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_id(self.domain, "DomainSpec.domain")
        if not isinstance(self.description, str):
            raise ValidationError("DomainSpec.description must be a string")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != DOMAIN_SCHEMA_VERSION
        ):
            raise ValidationError(
                f"unsupported schema version {self.schema_version!r}; "
                f"this build defines {DOMAIN_SCHEMA_VERSION}"
            )
        for field_name in ("slots", "intents"):
            value = getattr(self, field_name)
            if isinstance(value, list):
                object.__setattr__(self, field_name, tuple(value))
            if not isinstance(getattr(self, field_name), tuple):
                raise ValidationError(f"DomainSpec.{field_name} must be a tuple/list")
        if not self.intents:
            raise ValidationError(f"domain {self.domain!r} must define at least one intent")
        slot_names: set[str] = set()
        for spec in self.slots:
            if not isinstance(spec, SlotTypeSpec):
                raise ValidationError("DomainSpec.slots must contain SlotTypeSpec instances")
            if spec.name in slot_names:
                raise ValidationError(f"duplicate slot {spec.name!r} in domain {self.domain!r}")
            slot_names.add(spec.name)
        intent_names: set[str] = set()
        for intent in self.intents:
            if not isinstance(intent, IntentDefinition):
                raise ValidationError("DomainSpec.intents must contain IntentDefinition instances")
            if intent.name in intent_names:
                raise ValidationError(f"duplicate intent {intent.name!r} in domain {self.domain!r}")
            intent_names.add(intent.name)
            for ref in intent.slots:
                if ref.slot not in slot_names:
                    raise ValidationError(
                        f"intent {intent.name!r} references unknown slot {ref.slot!r} "
                        f"in domain {self.domain!r}"
                    )

    @property
    def slot_names(self) -> tuple[str, ...]:
        """All slot names, in declaration order."""
        return tuple(spec.name for spec in self.slots)

    @property
    def intent_names(self) -> tuple[str, ...]:
        """All intent names, in declaration order."""
        return tuple(intent.name for intent in self.intents)

    def slot_spec(self, name: str) -> SlotTypeSpec:
        """Look up a slot by name (strict)."""
        for spec in self.slots:
            if spec.name == name:
                return spec
        raise ValidationError(f"domain {self.domain!r} has no slot {name!r}")

    def intent(self, name: str) -> IntentDefinition:
        """Look up an intent by name (strict)."""
        for intent in self.intents:
            if intent.name == name:
                return intent
        raise ValidationError(f"domain {self.domain!r} has no intent {name!r}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-native dict form under the domain schema envelope."""
        return {
            "schema": DOMAIN_SCHEMA_ID,
            "schema_version": self.schema_version,
            "domain": self.domain,
            "description": self.description,
            "slots": [spec.to_dict() for spec in self.slots],
            "intents": [intent.to_dict() for intent in self.intents],
        }

    @classmethod
    def from_dict(cls, data: Any) -> DomainSpec:
        """Reconstruct a domain, rejecting unknown keys, ids, and versions."""
        if not isinstance(data, dict):
            raise SchemaError(f"domain payload must be a dict, got {type(data).__name__}")
        required = {"schema", "schema_version", "domain", "description", "slots", "intents"}
        unknown = sorted(set(data) - required)
        missing = sorted(required - set(data))
        if unknown:
            raise SchemaError(f"domain dict got unknown keys: {unknown}")
        if missing:
            raise SchemaError(f"domain dict is missing keys: {missing}")
        if data["schema"] != DOMAIN_SCHEMA_ID:
            raise SchemaError(
                f"unknown schema id {data['schema']!r}; expected {DOMAIN_SCHEMA_ID!r}"
            )
        if data["schema_version"] != DOMAIN_SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported schema version {data['schema_version']!r}; "
                f"this build parses {DOMAIN_SCHEMA_VERSION}"
            )
        if not isinstance(data["slots"], list) or not isinstance(data["intents"], list):
            raise SchemaError("domain 'slots' and 'intents' must be lists")
        return cls(
            domain=data["domain"],
            slots=tuple(SlotTypeSpec.from_dict(item) for item in data["slots"]),
            intents=tuple(IntentDefinition.from_dict(item) for item in data["intents"]),
            description=data["description"],
        )

    @property
    def domain_hash(self) -> str:
        """Provenance hash of the full definition (canonical JSON, SHA-256)."""
        return stable_hash(self.to_dict())
