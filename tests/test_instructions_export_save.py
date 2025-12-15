"""Export layout, golden manifest keys, and statistics correctness."""

import re

import pytest

from slotvox.config import GenerationConfig
from slotvox.data.dataset import generate_dataset
from slotvox.errors import ValidationError
from slotvox.instructions.export import EXPORT_SCHEMA_ID, EXPORT_SCHEMA_VERSION, export_instructions
from slotvox.instructions.templates import render_sample
from slotvox.schema.serialize import read_json

GOLDEN_MANIFEST_KEYS = [
    "content_hash",
    "dataset_hashes",
    "domains",
    "intents",
    "languages",
    "mixing_weights",
    "quality",
    "sample_count",
    "schema",
    "schema_version",
    "total_duration_ms",
]


def build_corpus():
    data = generate_dataset(
        GenerationConfig(domain="weather", seed=29, counts={"train": 3}, n_speakers=1)
    )
    samples = [
        render_sample(example, domain="weather", dataset_hash=data.dataset_hash)
        for example in data.examples
    ]
    return data, samples


def test_layout_and_golden_manifest(tmp_path):
    data, samples = build_corpus()
    root = export_instructions(samples, tmp_path / "corpus", mixing_weights={"weather": 1.0})
    assert (root / "instructions.jsonl").is_file()
    assert (root / "manifest.json").is_file()
    manifest = read_json(root / "manifest.json")
    assert sorted(manifest) == GOLDEN_MANIFEST_KEYS
    assert manifest["schema"] == EXPORT_SCHEMA_ID
    assert manifest["schema_version"] == EXPORT_SCHEMA_VERSION
    assert manifest["sample_count"] == 3
    assert manifest["languages"] == {"zh": 3}
    assert manifest["domains"] == {"weather": 3}
    assert sum(manifest["intents"].values()) == 3
    assert manifest["dataset_hashes"] == [data.dataset_hash]
    assert manifest["mixing_weights"] == {"weather": 1.0}
    assert manifest["quality"]["passed"] == 3
    assert manifest["quality"]["failed"] == 0
    assert manifest["total_duration_ms"] > 0
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["content_hash"])


def test_export_rejects_empty_and_duplicate_ids(tmp_path):
    _, samples = build_corpus()
    with pytest.raises(ValidationError, match="non-empty"):
        export_instructions((), tmp_path / "x")
    with pytest.raises(ValidationError, match="duplicate"):
        export_instructions([samples[0], samples[0]], tmp_path / "y")
    with pytest.raises(ValidationError, match="InstructionSample"):
        export_instructions(["nope"], tmp_path / "z")


def test_export_validates_weights_and_overwrite(tmp_path):
    _, samples = build_corpus()
    with pytest.raises(ValidationError, match="weight"):
        export_instructions(samples, tmp_path / "w", mixing_weights={"a": -1.0})
    root = export_instructions(samples, tmp_path / "corpus")
    with pytest.raises(ValidationError, match="overwrite"):
        export_instructions(samples, root)
    export_instructions(samples, root, overwrite=True)
