"""Golden pin for the data package API surface."""

import numpy as np

from slotvox import data
from slotvox.config.generation import GenerationConfig

GOLDEN_ALL = [
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


def test_data_api_surface_is_pinned():
    assert sorted(data.__all__) == GOLDEN_ALL
    for name in GOLDEN_ALL:
        assert hasattr(data, name)


def test_persist_round_trip_smoke(tmp_path):
    dataset = data.generate_dataset(
        GenerationConfig(domain="weather", seed=7, counts={"train": 2}, n_speakers=1)
    )
    root = data.save_dataset(dataset, tmp_path / "ds")
    loaded = data.load_dataset(root)
    assert loaded.dataset_hash == dataset.dataset_hash
    assert data.summarize(loaded)["counts"]["train"] == 2


def test_feature_store_smoke(tmp_path):
    from slotvox.config import FeatureConfig

    dataset = data.generate_dataset(
        GenerationConfig(domain="weather", seed=7, counts={"train": 2}, n_speakers=1)
    )
    root = data.featurize_dataset(dataset, FeatureConfig(n_mels=8), tmp_path / "feat")
    ids = data.list_features(root)
    mel, token_map = data.load_features(root, ids[0])
    assert mel.shape[1] == 8
    assert mel.shape[0] == token_map.shape[0]
    assert np.any(token_map >= 0)  # token frames exist alongside gaps
    assert np.all(token_map >= data.NO_TOKEN)
