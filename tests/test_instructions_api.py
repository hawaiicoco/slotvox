"""Golden pin for the instructions package API surface."""

import slotvox.instructions as instructions

GOLDEN_ALL = [
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


def test_api_surface_is_pinned():
    assert sorted(instructions.__all__) == GOLDEN_ALL
    for name in GOLDEN_ALL:
        assert hasattr(instructions, name)


def test_schema_constants():
    assert instructions.INSTRUCTION_SCHEMA_ID == "slotvox.instruction-sample"
    assert instructions.INSTRUCTION_SCHEMA_VERSION == 1
    assert instructions.TURN_ROLES == ("system", "user", "assistant")
