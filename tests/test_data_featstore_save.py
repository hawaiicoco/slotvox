"""featurize_dataset layout and envelope contracts."""

import pytest

from slotvox.config import FeatureConfig, GenerationConfig
from slotvox.data.dataset import generate_dataset
from slotvox.data.features_store import featurize_dataset
from slotvox.errors import ValidationError
from slotvox.schema.serialize import read_json


def build_dataset():
    return generate_dataset(
        GenerationConfig(domain="weather", seed=31, counts={"train": 2}, n_speakers=1)
    )


def test_layout_and_golden_envelope_keys(tmp_path):
    dataset = build_dataset()
    config = FeatureConfig(n_mels=8)
    root = featurize_dataset(dataset, config, tmp_path / "feat")
    envelope = read_json(root / "features.json")
    assert sorted(envelope) == [
        "dataset_hash",
        "example_count",
        "feature_config",
        "sample_rate",
        "schema",
        "schema_version",
    ]
    assert envelope["schema"] == "slotvox.features"
    assert envelope["example_count"] == 2
    assert envelope["dataset_hash"] == dataset.dataset_hash
    arrays = sorted(path.stem for path in root.glob("*.npz"))
    assert arrays == sorted(e.utterance_id for e in dataset.examples)


def test_overwrite_guard(tmp_path):
    dataset = build_dataset()
    featurize_dataset(dataset, FeatureConfig(n_mels=8), tmp_path / "feat")
    with pytest.raises(ValidationError, match="overwrite=True"):
        featurize_dataset(dataset, FeatureConfig(n_mels=8), tmp_path / "feat")


def test_type_rejections(tmp_path):
    with pytest.raises(ValidationError, match="GeneratedDataset"):
        featurize_dataset("x", FeatureConfig(), tmp_path / "f")
    with pytest.raises(ValidationError, match="FeatureConfig"):
        featurize_dataset(build_dataset(), "x", tmp_path / "f")
