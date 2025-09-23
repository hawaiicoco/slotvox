"""load_dataset contracts: exact labels, close audio, corruption guards."""

import json

import numpy as np
import pytest

from slotvox.config.generation import GenerationConfig
from slotvox.data.dataset import generate_dataset
from slotvox.data.persist import load_dataset, save_dataset
from slotvox.errors import AudioError, SchemaError
from slotvox.schema.serialize import read_jsonl, write_jsonl


def build_dataset():
    config = GenerationConfig(
        domain="weather", seed=21, counts={"train": 3, "dev": 1}, n_speakers=2
    )
    return generate_dataset(config)


def saved(tmp_path):
    dataset = build_dataset()
    root = save_dataset(dataset, tmp_path / "ds")
    return dataset, root


def test_round_trip_labels_exact_audio_close(tmp_path):
    original, root = saved(tmp_path)
    loaded = load_dataset(root)
    assert loaded.dataset_hash == original.dataset_hash
    assert [e.annotation for e in loaded.examples] == [e.annotation for e in original.examples]
    assert [e.slot_values for e in loaded.examples] == [e.slot_values for e in original.examples]
    for got, want in zip(loaded.examples, original.examples, strict=True):
        assert got.utterance.segments == want.utterance.segments
        assert np.allclose(got.utterance.samples, want.utterance.samples, atol=1e-4)


def test_missing_files_rejected(tmp_path):
    with pytest.raises(SchemaError, match="not found"):
        load_dataset(tmp_path / "nowhere")
    _, root = saved(tmp_path)
    (root / "manifest.jsonl").unlink()
    with pytest.raises(SchemaError, match="not found"):
        load_dataset(root)


def test_tampered_manifest_rejected_by_hash(tmp_path):
    _, root = saved(tmp_path)
    rows = read_jsonl(root / "manifest.jsonl")
    rows[0]["speaker_id"] = rows[0]["speaker_id"] + 97  # structurally valid, wrong content
    write_jsonl(root / "manifest.jsonl", rows)
    with pytest.raises(SchemaError, match="hash mismatch"):
        load_dataset(root)


def test_missing_wav_rejected(tmp_path):
    _, root = saved(tmp_path)
    next((root / "audio").glob("*.wav")).unlink()
    with pytest.raises(AudioError, match="not found"):
        load_dataset(root)


def test_bad_envelope_rejected(tmp_path):
    dataset = build_dataset()
    root = save_dataset(dataset, tmp_path / "ds")
    envelope_path = root / "dataset.json"
    envelope = json.loads(envelope_path.read_text())
    envelope["schema_version"] = 99
    envelope_path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(SchemaError, match="schema version"):
        load_dataset(root)
