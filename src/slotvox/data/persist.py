"""On-disk dataset format: envelope JSON + JSONL manifest + WAV audio.

Layout (schema ``slotvox.dataset``, version 1)::

    <dir>/dataset.json      envelope: schema, provenance, stats, hash
    <dir>/manifest.jsonl    one GeneratedExample.to_dict() row per example
    <dir>/audio/<id>.wav    mono PCM16 audio per utterance

The envelope is written last and acts as the commit point: a directory
without ``dataset.json`` is an incomplete write, never a valid dataset.
"""

from __future__ import annotations

from pathlib import Path

from slotvox.audio.riff import write_wav
from slotvox.data.dataset import GeneratedDataset
from slotvox.data.stats import summarize
from slotvox.errors import ValidationError
from slotvox.schema.serialize import write_json_atomic, write_jsonl

DATASET_DIR_SCHEMA_ID = "slotvox.dataset"
DATASET_DIR_SCHEMA_VERSION = 1


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
