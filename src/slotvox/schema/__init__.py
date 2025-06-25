"""Task schema: versioned, strictly validated definitions.

The schema layer defines what a spoken language understanding task *is*:
slot types with value constraints, intents with slot inventories, and the
domains that bind them together. Every definition validates strictly in
both directions (construct/parse) and round-trips through canonical JSON.
"""

from slotvox.schema.intents import IntentDefinition, SlotRef
from slotvox.schema.naming import ID_PATTERN, MAX_ID_LENGTH, validate_id
from slotvox.schema.slots import SLOT_VALUE_TYPES, SlotTypeSpec

__all__ = [
    "ID_PATTERN",
    "IntentDefinition",
    "MAX_ID_LENGTH",
    "SLOT_VALUE_TYPES",
    "SlotRef",
    "SlotTypeSpec",
    "validate_id",
]
