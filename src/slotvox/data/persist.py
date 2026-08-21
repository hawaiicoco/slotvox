"""On-disk dataset format: envelope JSON + JSONL manifest + WAV audio.

Layout (schema ``slotvox.dataset``, version 1)::

    <dir>/dataset.json      envelope: schema, provenance, stats, hash
    <dir>/manifest.jsonl    one GeneratedExample.to_dict() row per example
    <dir>/audio/<id>.wav    mono PCM16 audio per utterance

The envelope is written last and acts as the commit point: a directory
without ``dataset.json`` is an incomplete write, never a valid dataset.
Loading revalidates everything — strict schemas and keys, per-split counts,
and a recomputed dataset hash that must equal the envelope's (a corruption
and tamper guard). Audio round-trips through PCM16, so samples match to
quantization tolerance while labels match exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from slotvox.audio.riff import read_wav, write_wav
from slotvox.config.base import config_from_dict
from slotvox.config.generation import GenerationConfig
from slotvox.data.dataset import GeneratedDataset
from slotvox.data.factory import GeneratedExample
from slotvox.data.stats import summarize
from slotvox.errors import SchemaError, ValidationError
from slotvox.schema.annotations import AnnotatedUtterance
from slotvox.schema.serialize import read_json, read_jsonl, write_json_atomic, write_jsonl
from slotvox.synth.utterance import SyntheticUtterance

DATASET_DIR_SCHEMA_ID = "slotvox.dataset"
DATASET_DIR_SCHEMA_VERSION = 1

_ROW_KEYS = {
    "utterance_id",
    "split",
    "speaker_id",
    "snr_db",
    "pattern_id",
    "annotation",
    "slot_values",
    "sample_rate",
    "n_samples",
    "segments",
}


def save_dataset(
    dataset: GeneratedDataset, out_dir: str | Path, *, overwrite: bool = False
) -> Path:
    """Write ``dataset`` to ``out_dir`` in the slotvox.dataset format."""
    if not isinstance(dataset, GeneratedDataset):
        raise ValidationError(
            f"save_dataset expects a GeneratedDataset, got {type(dataset).__name__}"
        )
    root = Path(out_dir)
    if (root / "dataset.json").exists() and not overwrite:
        raise ValidationError(f"{root} already holds a dataset; pass overwrite=True to replace")
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for example in dataset.examples:
        write_wav(audio_dir / f"{example.utterance_id}.wav", example.utterance.audio)
        rows.append(example.to_dict())
    write_jsonl(root / "manifest.jsonl", rows)
    envelope = {
        "schema": DATASET_DIR_SCHEMA_ID,
        "schema_version": DATASET_DIR_SCHEMA_VERSION,
        "example_count": len(rows),
        "provenance": dataset.provenance(),
        "stats": summarize(dataset),
        "dataset_hash": dataset.dataset_hash,
    }
    write_json_atomic(root / "dataset.json", envelope)
    return root


def _example_from_row(row: Any, root: Path, index: int) -> GeneratedExample:
    if not isinstance(row, dict):
        raise SchemaError(f"manifest row {index}: expected an object")
    unknown = sorted(set(row) - _ROW_KEYS)
    missing = sorted(_ROW_KEYS - set(row))
    if unknown:
        raise SchemaError(f"manifest row {index} got unknown keys: {unknown}")
    if missing:
        raise SchemaError(f"manifest row {index} is missing keys: {missing}")
    audio = read_wav(root / "audio" / f"{row['utterance_id']}.wav")
    if audio.sample_rate != row["sample_rate"]:
        raise SchemaError(f"manifest row {index}: wav sample rate disagrees with manifest")
    utterance = SyntheticUtterance.from_dict(
        {
            "sample_rate": row["sample_rate"],
            "n_samples": row["n_samples"],
            "snr_db": row["snr_db"],
            "segments": row["segments"],
        },
        audio.samples,
    )
    return GeneratedExample(
        utterance_id=row["utterance_id"],
        split=row["split"],
        speaker_id=row["speaker_id"],
        snr_db=row["snr_db"],
        pattern_id=row["pattern_id"],
        annotation=AnnotatedUtterance.from_dict(row["annotation"]),
        utterance=utterance,
        slot_values={slot: tuple(values) for slot, values in row["slot_values"].items()},
    )


def load_dataset(in_dir: str | Path) -> GeneratedDataset:
    """Read and revalidate a dataset directory (strict, both directions)."""
    root = Path(in_dir)
    envelope = read_json(root / "dataset.json")
    if not isinstance(envelope, dict):
        raise SchemaError("dataset.json must contain a JSON object")
    if envelope.get("schema") != DATASET_DIR_SCHEMA_ID:
        raise SchemaError(f"unknown dataset schema {envelope.get('schema')!r}")
    if envelope.get("schema_version") != DATASET_DIR_SCHEMA_VERSION:
        raise SchemaError(f"unsupported dataset schema version {envelope.get('schema_version')!r}")
    rows = read_jsonl(root / "manifest.jsonl")
    if len(rows) != envelope.get("example_count"):
        raise SchemaError(
            f"manifest has {len(rows)} rows, envelope says {envelope.get('example_count')}"
        )
    examples = tuple(_example_from_row(row, root, index) for index, row in enumerate(rows))
    provenance = envelope.get("provenance")
    if not isinstance(provenance, dict):
        raise SchemaError("dataset envelope has no provenance object")
    domain_id = provenance.get("domain_id")
    language = provenance.get("language")
    policy = provenance.get("policy")
    seed = provenance.get("seed")
    config_payload = provenance.get("config")
    if not isinstance(domain_id, str):
        raise SchemaError("provenance domain_id must be a string")
    if not isinstance(language, str):
        raise SchemaError("provenance language must be a string")
    if not isinstance(policy, str):
        raise SchemaError("provenance policy must be a string")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SchemaError("provenance seed must be an int")
    if not isinstance(config_payload, dict):
        raise SchemaError("provenance config must be a dict")
    config = config_from_dict(config_payload)
    if not isinstance(config, GenerationConfig):
        raise SchemaError("provenance config must describe a GenerationConfig")
    dataset = GeneratedDataset(
        domain_id=domain_id,
        language=language,
        policy=policy,
        seed=seed,
        config=config,
        examples=examples,
    )
    if dataset.dataset_hash != envelope.get("dataset_hash"):
        raise SchemaError("dataset hash mismatch: manifest or envelope corrupted")
    return dataset
