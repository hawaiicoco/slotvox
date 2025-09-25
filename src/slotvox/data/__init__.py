"""Synthetic dialogue factory, splits, statistics, and their vocabularies."""

from slotvox.data.acoustics import speaker_shift, token_segment_plan
from slotvox.data.alignment import NO_TOKEN, frame_tags, frame_token_map
from slotvox.data.dataset import GeneratedDataset, generate_dataset
from slotvox.data.factory import GeneratedExample, instantiate_template, render_example
from slotvox.data.features_store import featurize_dataset, list_features, load_features
from slotvox.data.lexicon import LexiconEntry, registered_lexicons, slot_entries
from slotvox.data.patterns import (
    PatternTemplate,
    TokenPlan,
    registered_templates,
    templates_for,
)
from slotvox.data.persist import load_dataset, save_dataset
from slotvox.data.splits import (
    SPLIT_POLICIES,
    assign_keys_to_splits,
    pattern_splits,
    speaker_splits,
)
from slotvox.data.stats import summarize

__all__ = [
    "GeneratedDataset",
    "GeneratedExample",
    "LexiconEntry",
    "NO_TOKEN",
    "PatternTemplate",
    "SPLIT_POLICIES",
    "TokenPlan",
    "assign_keys_to_splits",
    "featurize_dataset",
    "frame_tags",
    "frame_token_map",
    "generate_dataset",
    "instantiate_template",
    "list_features",
    "load_dataset",
    "load_features",
    "pattern_splits",
    "registered_lexicons",
    "registered_templates",
    "render_example",
    "save_dataset",
    "slot_entries",
    "speaker_shift",
    "speaker_splits",
    "summarize",
    "templates_for",
    "token_segment_plan",
]
