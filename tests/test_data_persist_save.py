"""save_dataset layout, envelope, and overwrite contracts."""

import pytest

from slotvox.config.generation import GenerationConfig
from slotvox.data.dataset import generate_dataset
from slotvox.data.persist import save_dataset
from slotvox.errors import ValidationError
from slotvox.schema.serialize import read_json, read_jsonl


def build_dataset():
    config = GenerationConfig(
        domain="weather", seed=21, counts={"train": 3, "dev": 1}, n_speakers=2
    )
    return generate_dataset(config)


def test_save_layout_and_golden_envelope_keys(tmp_path):
    dataset = build_dataset()
    root = save_dataset(dataset, tmp_path / "ds")
    assert (root / "dataset.json").is_file()
    assert (root / "manifest.jsonl").is_file()
    envelope = read_json(root / "dataset.json")
    assert sorted(envelope) == [
        "dataset_hash",
        "example_count",
        "provenance",
        "schema",
        "schema_version",
        "stats",
    ]
    assert envelope["schema"] == "slotvox.dataset"
    assert envelope["schema_version"] == 1
    assert envelope["example_count"] == 4
    assert envelope["dataset_hash"] == dataset.dataset_hash
    rows = read_jsonl(root / "manifest.jsonl")
    assert len(rows) == 4
    wavs = sorted((root / "audio").glob("*.wav"))
    assert [path.stem for path in wavs] == sorted(row["utterance_id"] for row in rows)


def test_save_refuses_overwrite_unless_asked(tmp_path):
    dataset = build_dataset()
    save_dataset(dataset, tmp_path / "ds")
    with pytest.raises(ValidationError, match="overwrite=True"):
        save_dataset(dataset, tmp_path / "ds")
    save_dataset(dataset, tmp_path / "ds", overwrite=True)


def test_save_rejects_non_datasets(tmp_path):
    with pytest.raises(ValidationError, match="GeneratedDataset"):
        save_dataset("not-a-dataset", tmp_path / "ds")
