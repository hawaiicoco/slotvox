"""Synthetic dialogue factory, splits, statistics, and their vocabularies."""

from slotvox.data.acoustics import speaker_shift, token_segment_plan
from slotvox.data.dataset import GeneratedDataset, generate_dataset
from slotvox.data.factory import GeneratedExample, instantiate_template, render_example
from slotvox.data.lexicon import LexiconEntry, registered_lexicons, slot_entries
from slotvox.data.patterns import PatternTemplate, TokenPlan, registered_templates, templates_for
from slotvox.data.splits import (
    SPLIT_POLICIES,
    assign_keys_to_splits,
    pattern_splits,
    speaker_splits,
)
from slotvox.data.stats import summarize

__all__ = [
    "SPLIT_POLICIES",
    "GeneratedDataset",
    "GeneratedExample",
    "LexiconEntry",
    "PatternTemplate",
    "TokenPlan",
    "assign_keys_to_splits",
    "generate_dataset",
    "instantiate_template",
    "pattern_splits",
    "registered_lexicons",
    "registered_templates",
    "render_example",
    "slot_entries",
    "speaker_shift",
    "speaker_splits",
    "summarize",
    "templates_for",
    "token_segment_plan",
]
