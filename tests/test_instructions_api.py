"""Golden pin for the instructions package API surface."""

import slotvox.instructions as instructions

GOLDEN_ALL = [
    "ASSISTANT_TEMPLATES",
    "AudioRef",
    "EMPTY_SLOTS",
    "EXPORT_SCHEMA_ID",
    "EXPORT_SCHEMA_VERSION",
    "INSTRUCTION_SCHEMA_ID",
    "INSTRUCTION_SCHEMA_VERSION",
    "InstructionSample",
    "NEAR_DUPLICATE_THRESHOLD",
    "QualityFlags",
    "QualityReport",
    "QualityThresholds",
    "SYSTEM_TEMPLATES",
    "TURN_ROLES",
    "Turn",
    "TurnTemplate",
    "canonical_answer_prefix",
    "check_provenance_hash",
    "check_sample",
    "check_samples",
    "cosine_similarity",
    "exact_duplicate_groups",
    "export_instructions",
    "feature_vector",
    "load_instructions",
    "mix_samples",
    "mixing_quotas",
    "near_duplicate_pairs",
    "normalize_text",
    "read_export_manifest",
    "render_sample",
    "sample_text",
    "slots_text",
    "template_variables",
    "text_hash",
]


def test_api_surface_is_pinned():
    assert sorted(instructions.__all__) == GOLDEN_ALL
    for name in GOLDEN_ALL:
        assert hasattr(instructions, name)


def test_schema_constants():
    assert instructions.INSTRUCTION_SCHEMA_ID == "slotvox.instruction-sample"
    assert instructions.INSTRUCTION_SCHEMA_VERSION == 1
    assert instructions.EXPORT_SCHEMA_ID == "slotvox.instruction-export"
    assert instructions.EXPORT_SCHEMA_VERSION == 1
    assert instructions.TURN_ROLES == ("system", "user", "assistant")
    assert instructions.NEAR_DUPLICATE_THRESHOLD == 0.9
