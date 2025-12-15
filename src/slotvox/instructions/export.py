"""Instruction-corpus export: JSONL rows plus a versioned manifest.

Layout (schema ``slotvox.instruction-export``, version 1)::

    <dir>/instructions.jsonl   one sample dict per line (canonical JSON)
    <dir>/manifest.json        envelope: counts, statistics, quality
                               summary, mixing weights, corpus hash

No audio bytes are stored — rows reference manifest ids that consumers
resolve against a dataset or feature store. The manifest is written last
and acts as the commit point; loading re-checks the row count and corpus
hash, so tampered or truncated exports are rejected before parsing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from slotvox.errors import SchemaError, ValidationError
from slotvox.instructions.quality import QualityThresholds, check_samples
from slotvox.instructions.schema import InstructionSample
from slotvox.schema.naming import validate_id
from slotvox.schema.serialize import (
    read_json,
    read_jsonl,
    write_json_atomic,
    write_jsonl,
)
from slotvox.util.jsoncanon import stable_hash

EXPORT_SCHEMA_ID = "slotvox.instruction-export"
EXPORT_SCHEMA_VERSION = 1


def _count_by(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def export_instructions(
    samples,
    out_dir,
    *,
    mixing_weights: Mapping[str, float] | None = None,
    snr_db: Mapping[str, float] | None = None,
    thresholds: QualityThresholds | None = None,
    overwrite: bool = False,
) -> Path:
    """Write ``instructions.jsonl`` + ``manifest.json`` under ``out_dir``."""
    if isinstance(samples, (str, bytes)) or not isinstance(samples, (list, tuple)) or not samples:
        raise ValidationError("export_instructions needs a non-empty list/tuple of samples")
    items: list[InstructionSample] = []
    seen: set[str] = set()
    for sample in samples:
        if not isinstance(sample, InstructionSample):
            raise ValidationError(
                f"samples must contain InstructionSample, got {type(sample).__name__}"
            )
        if sample.sample_id in seen:
            raise ValidationError(f"duplicate sample_id {sample.sample_id!r}")
        seen.add(sample.sample_id)
        items.append(sample)
    weights_payload = None
    if mixing_weights is not None:
        if not isinstance(mixing_weights, Mapping) or not mixing_weights:
            raise ValidationError("mixing_weights must be a non-empty mapping or None")
        checked: dict[str, float] = {}
        for source, weight in mixing_weights.items():
            validate_id(source, "mixing weight source")
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight < 0.0
            ):
                raise ValidationError(
                    f"mixing weight for {source!r} must be a finite number >= 0, got {weight!r}"
                )
            checked[source] = float(weight)
        weights_payload = {key: checked[key] for key in sorted(checked)}
    root = Path(out_dir)
    if (root / "manifest.json").exists() and not overwrite:
        raise ValidationError(f"{root} already holds an export; pass overwrite=True to replace")
    root.mkdir(parents=True, exist_ok=True)
    rows = [sample.to_dict() for sample in items]
    write_jsonl(root / "instructions.jsonl", rows)
    report = check_samples(items, snr_db=snr_db, thresholds=thresholds)
    refs = [turn.audio for sample in items for turn in sample.turns if turn.audio is not None]
    manifest = {
        "schema": EXPORT_SCHEMA_ID,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "sample_count": len(items),
        "languages": _count_by(sample.language for sample in items),
        "domains": _count_by(sample.domain for sample in items),
        "intents": _count_by(sample.intent for sample in items),
        "total_duration_ms": float(sum(ref.duration_ms for ref in refs)),
        "dataset_hashes": sorted({ref.dataset_hash for ref in refs}),
        "mixing_weights": weights_payload,
        "quality": report.to_dict(),
        "content_hash": stable_hash(rows),
    }
    write_json_atomic(root / "manifest.json", manifest)
    return root


_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "sample_count",
        "languages",
        "domains",
        "intents",
        "total_duration_ms",
        "dataset_hashes",
        "mixing_weights",
        "quality",
        "content_hash",
    }
)


def read_export_manifest(in_dir: str | Path) -> dict:
    """Read and validate the export manifest envelope (strict)."""
    manifest = read_json(Path(in_dir) / "manifest.json")
    if not isinstance(manifest, dict):
        raise SchemaError("manifest.json must contain a JSON object")
    unknown = sorted(set(manifest) - _MANIFEST_KEYS)
    missing = sorted(_MANIFEST_KEYS - set(manifest))
    if unknown:
        raise SchemaError(f"export manifest got unknown keys: {unknown}")
    if missing:
        raise SchemaError(f"export manifest is missing keys: {missing}")
    if manifest["schema"] != EXPORT_SCHEMA_ID:
        raise SchemaError(f"unknown export schema {manifest['schema']!r}")
    if manifest["schema_version"] != EXPORT_SCHEMA_VERSION:
        raise SchemaError(f"unsupported export schema version {manifest['schema_version']!r}")
    return manifest


def load_instructions(in_dir: str | Path) -> tuple[InstructionSample, ...]:
    """Load an exported corpus, rejecting truncation and tampering."""
    manifest = read_export_manifest(in_dir)
    rows = read_jsonl(Path(in_dir) / "instructions.jsonl")
    if len(rows) != manifest["sample_count"]:
        raise SchemaError(
            f"instruction row count {len(rows)} does not match manifest "
            f"sample_count {manifest['sample_count']}"
        )
    if stable_hash(rows) != manifest["content_hash"]:
        raise SchemaError("instructions are corrupt or tampered: content hash mismatch")
    return tuple(InstructionSample.from_dict(row) for row in rows)
