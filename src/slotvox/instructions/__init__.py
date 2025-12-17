"""Instruction-data pipeline for speech LLMs (torch-free, fully offline).

Schemas and rendering (speech-text turns referencing audio by manifest
id), deterministic weighted mixing, near-duplicate detection, quality
flags, and versioned corpus export. The pipeline produces DATA for
documented training protocols; slotvox ships no model weights and
downloads nothing.
"""

from slotvox.instructions.dedup import (
    NEAR_DUPLICATE_THRESHOLD,
    cosine_similarity,
    exact_duplicate_groups,
    feature_vector,
    near_duplicate_pairs,
    normalize_text,
    sample_text,
    text_hash,
)
from slotvox.instructions.export import (
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
    export_instructions,
    load_instructions,
    read_export_manifest,
)
from slotvox.instructions.mixing import mix_samples, mixing_quotas
from slotvox.instructions.quality import (
    QualityFlags,
    QualityReport,
    QualityThresholds,
    canonical_answer_prefix,
    check_sample,
    check_samples,
)
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
