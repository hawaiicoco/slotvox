"""Intent definitions: labeled intents with their slot inventories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from slotvox.errors import ValidationError
from slotvox.schema.naming import validate_id


@dataclass(frozen=True)
class SlotRef:
    """A reference to a domain slot, with optionality."""

    slot: str
    optional: bool = False

    def __post_init__(self) -> None:
        validate_id(self.slot, "SlotRef.slot")
        if not isinstance(self.optional, bool):
            raise ValidationError(f"SlotRef.optional must be a bool, got {self.optional!r}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-native dict form."""
        return {"slot": self.slot, "optional": self.optional}

    @classmethod
    def from_dict(cls, data: Any) -> SlotRef:
        """Reconstruct a reference, rejecting unknown or missing keys."""
        if not isinstance(data, dict):
            raise ValidationError(f"slot ref payload must be a dict, got {type(data).__name__}")
        required = {"slot", "optional"}
        unknown = sorted(set(data) - required)
        missing = sorted(required - set(data))
        if unknown:
            raise ValidationError(f"slot ref dict got unknown keys: {unknown}")
        if missing:
            raise ValidationError(f"slot ref dict is missing keys: {missing}")
        return cls(slot=data["slot"], optional=data["optional"])


@dataclass(frozen=True)
class IntentDefinition:
    """One intent of a domain: an id plus the slots it can take."""

    name: str
    slots: tuple[SlotRef, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        validate_id(self.name, "IntentDefinition.name")
        if not isinstance(self.description, str):
            raise ValidationError("IntentDefinition.description must be a string")
        slots = self.slots
        if isinstance(slots, list):
            slots = tuple(slots)
            object.__setattr__(self, "slots", slots)
        if not isinstance(slots, tuple):
            raise ValidationError("IntentDefinition.slots must be a tuple/list of SlotRef")
        seen: set[str] = set()
        for ref in slots:
            if not isinstance(ref, SlotRef):
                raise ValidationError("IntentDefinition.slots must contain SlotRef instances")
            if ref.slot in seen:
                raise ValidationError(f"intent {self.name!r} references slot {ref.slot!r} twice")
            seen.add(ref.slot)

    @property
    def slot_names(self) -> tuple[str, ...]:
        """All referenced slot names, in declaration order."""
        return tuple(ref.slot for ref in self.slots)

    @property
    def required_slots(self) -> tuple[str, ...]:
        """Non-optional slot names, in declaration order."""
        return tuple(ref.slot for ref in self.slots if not ref.optional)
