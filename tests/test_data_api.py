"""Golden pin for the data package API surface."""

from slotvox import data
from slotvox.config.generation import GenerationConfig

GOLDEN_ALL = [
    "GeneratedDataset",
    "GeneratedExample",
    "LexiconEntry",
    "PatternTemplate",
    "SPLIT_POLICIES",
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


def test_data_api_surface_is_pinned():
    assert sorted(data.__all__) == GOLDEN_ALL
    for name in GOLDEN_ALL:
        assert hasattr(data, name)


def test_pipeline_smoke():
    dataset = data.generate_dataset(
        GenerationConfig(domain="weather", seed=7, counts={"train": 2}, n_speakers=1)
    )
    stats = data.summarize(dataset)
    assert stats["counts"]["train"] == 2
    assert len(dataset.examples) == 2
