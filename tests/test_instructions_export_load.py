"""Strict corpus loading: roundtrips, tamper and truncation rejection."""

import json

import pytest

from slotvox.config import GenerationConfig
from slotvox.data.dataset import generate_dataset
from slotvox.errors import SchemaError
from slotvox.instructions.export import (
    export_instructions,
    load_instructions,
    read_export_manifest,
)
from slotvox.instructions.templates import render_sample
from slotvox.schema.serialize import read_json, write_json_atomic


def build_corpus():
    data = generate_dataset(
        GenerationConfig(domain="weather", seed=31, counts={"train": 3}, n_speakers=1)
    )
    return [
        render_sample(example, domain="weather", dataset_hash=data.dataset_hash)
        for example in data.examples
    ]


def test_roundtrip_load(tmp_path):
    samples = build_corpus()
    root = export_instructions(samples, tmp_path / "corpus")
    assert load_instructions(root) == tuple(samples)


def test_tampered_row_rejected(tmp_path):
    samples = build_corpus()
    root = export_instructions(samples, tmp_path / "corpus")
    path = root / "instructions.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["intent"] = "weather-alert"
    lines[0] = json.dumps(row, ensure_ascii=False, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="tampered"):
        load_instructions(root)


def test_truncated_rows_rejected(tmp_path):
    samples = build_corpus()
    root = export_instructions(samples, tmp_path / "corpus")
    path = root / "instructions.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="does not match"):
        load_instructions(root)


def test_manifest_envelope_strict(tmp_path):
    samples = build_corpus()
    root = export_instructions(samples, tmp_path / "corpus")
    manifest = read_json(root / "manifest.json")
    write_json_atomic(root / "manifest.json", {**manifest, "schema": "other"})
    with pytest.raises(SchemaError, match="schema"):
        read_export_manifest(root)
    write_json_atomic(root / "manifest.json", {k: v for k, v in manifest.items() if k != "quality"})
    with pytest.raises(SchemaError, match="missing"):
        read_export_manifest(root)
    with pytest.raises(SchemaError, match="not found"):
        load_instructions(tmp_path / "missing")
