"""Task schema: versioned, strictly validated definitions.

The schema layer defines what a spoken language understanding task *is*:
slot types with value constraints, intents with slot inventories, the
domains that bind them together, utterance↔annotation pairs, and bounded
file helpers for schema artifacts. Every definition validates strictly in
both directions (construct/parse) and round-trips through canonical JSON.
"""

from slotvox.schema.annotations import LANGUAGES, AnnotatedUtterance
from slotvox.schema.builtins import (
    BUILTIN_DOMAIN_IDS,
    builtin_domain,
    builtin_domains,
    calendar_domain,
    music_control_domain,
    navigation_domain,
    weather_domain,
)
from slotvox.schema.domains import DOMAIN_SCHEMA_ID, DOMAIN_SCHEMA_VERSION, DomainSpec
from slotvox.schema.intents import IntentDefinition, SlotRef
from slotvox.schema.naming import ID_PATTERN, MAX_ID_LENGTH, validate_id
from slotvox.schema.serialize import (
    MAX_ARTIFACT_BYTES,
    iter_jsonl,
    read_json,
    read_jsonl,
    write_json_atomic,
    write_jsonl,
)
from slotvox.schema.slots import SLOT_VALUE_TYPES, SlotTypeSpec

__all__ = [
    "AnnotatedUtterance",
    "BUILTIN_DOMAIN_IDS",
    "DOMAIN_SCHEMA_ID",
    "DOMAIN_SCHEMA_VERSION",
    "DomainSpec",
    "ID_PATTERN",
    "IntentDefinition",
    "LANGUAGES",
    "MAX_ARTIFACT_BYTES",
    "MAX_ID_LENGTH",
    "SLOT_VALUE_TYPES",
    "SlotRef",
    "SlotTypeSpec",
    "builtin_domain",
    "builtin_domains",
    "calendar_domain",
    "iter_jsonl",
    "music_control_domain",
    "navigation_domain",
    "read_json",
    "read_jsonl",
    "validate_id",
    "weather_domain",
    "write_json_atomic",
    "write_jsonl",
]
