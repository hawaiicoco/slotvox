"""Instruction-data pipeline for speech LLMs: schemas, templates, rendering.

Everything here is torch-free and offline: the pipeline produces DATA
(JSON-native, schema-versioned) for documented training protocols. No
model weights ship with slotvox and nothing is downloaded.
"""

from slotvox.instructions.schema import (
    INSTRUCTION_SCHEMA_ID,
    INSTRUCTION_SCHEMA_VERSION,
    TURN_ROLES,
    AudioRef,
    InstructionSample,
    Turn,
    check_provenance_hash,
)
from slotvox.instructions.templates import (
    ASSISTANT_TEMPLATES,
    EMPTY_SLOTS,
    SYSTEM_TEMPLATES,
    TurnTemplate,
    render_sample,
    slots_text,
    template_variables,
)

__all__ = [
    "ASSISTANT_TEMPLATES",
    "AudioRef",
    "EMPTY_SLOTS",
    "INSTRUCTION_SCHEMA_ID",
    "INSTRUCTION_SCHEMA_VERSION",
    "InstructionSample",
    "SYSTEM_TEMPLATES",
    "TURN_ROLES",
    "Turn",
    "TurnTemplate",
    "check_provenance_hash",
    "render_sample",
    "slots_text",
    "template_variables",
]
