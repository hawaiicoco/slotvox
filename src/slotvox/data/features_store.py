"""Feature store: log-mel features and frame alignment per utterance.

Layout (schema ``slotvox.features``, version 1)::

    <dir>/features.json        envelope: feature config, dataset hash, count
    <dir>/<utterance_id>.npz   arrays: mel (float32 [T, n_mels]),
                               frame_token (int64 [T])

Features are derived data; the envelope records the source dataset hash so
consumers can refuse feature/label pairs that do not belong together. The
envelope is written last and acts as the commit point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from slotvox.config.features import FeatureConfig
from slotvox.data.alignment import frame_token_map
from slotvox.data.dataset import GeneratedDataset
from slotvox.errors import SchemaError, ValidationError
from slotvox.features.frontend import log_mel
from slotvox.schema.serialize import read_json, write_json_atomic

FEATURES_SCHEMA_ID = "slotvox.features"
FEATURES_SCHEMA_VERSION = 1
FEATURE_ARRAYS = ("frame_token", "mel")


def featurize_dataset(
    dataset: GeneratedDataset,
    feature_config: FeatureConfig,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Compute and store log-mel features plus frame alignment per example."""
    if not isinstance(dataset, GeneratedDataset):
        raise ValidationError(
            f"featurize_dataset expects a GeneratedDataset, got {type(dataset).__name__}"
        )
    if not isinstance(feature_config, FeatureConfig):
        raise ValidationError(
            f"feature_config must be a FeatureConfig, got {type(feature_config).__name__}"
        )
    root = Path(out_dir)
    if (root / "features.json").exists() and not overwrite:
        raise ValidationError(f"{root} already holds features; pass overwrite=True to replace")
    root.mkdir(parents=True, exist_ok=True)
    for example in dataset.examples:
        mel = log_mel(example.utterance.samples, feature_config)
        token_map = frame_token_map(example.utterance, feature_config)
        np.savez(root / f"{example.utterance_id}.npz", mel=mel, frame_token=token_map)
    envelope = {
        "schema": FEATURES_SCHEMA_ID,
        "schema_version": FEATURES_SCHEMA_VERSION,
        "feature_config": feature_config.to_dict(),
        "dataset_hash": dataset.dataset_hash,
        "sample_rate": feature_config.sample_rate,
        "example_count": len(dataset.examples),
    }
    write_json_atomic(root / "features.json", envelope)
    return root


def read_features_envelope(in_dir: str | Path) -> dict[str, Any]:
    """Read and validate the feature-store envelope (strict)."""
    envelope = read_json(Path(in_dir) / "features.json")
    if not isinstance(envelope, dict):
        raise SchemaError("features.json must contain a JSON object")
    if envelope.get("schema") != FEATURES_SCHEMA_ID:
        raise SchemaError(f"unknown features schema {envelope.get('schema')!r}")
    if envelope.get("schema_version") != FEATURES_SCHEMA_VERSION:
        raise SchemaError(f"unsupported features schema version {envelope.get('schema_version')!r}")
    return envelope


def list_features(in_dir: str | Path) -> tuple[str, ...]:
    """Sorted utterance ids with stored features; count must match envelope."""
    envelope = read_features_envelope(in_dir)
    ids = tuple(sorted(path.stem for path in Path(in_dir).glob("*.npz")))
    if len(ids) != envelope.get("example_count"):
        raise SchemaError(
            f"feature dir holds {len(ids)} arrays, envelope says {envelope.get('example_count')}"
        )
    return ids


def load_features(in_dir: str | Path, utterance_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(mel, frame_token)`` for one utterance, validating shapes.

    The utterance id is restricted to safe filename characters so a hostile
    id cannot escape the feature directory.
    """
    envelope = read_features_envelope(in_dir)
    if (
        not isinstance(utterance_id, str)
        or not utterance_id
        or any(ch.isspace() for ch in utterance_id)
        or "/" in utterance_id
        or "\\" in utterance_id
        or ".." in utterance_id
    ):
        raise SchemaError(f"invalid utterance id {utterance_id!r}")
    path = Path(in_dir) / f"{utterance_id}.npz"
    if not path.is_file():
        raise SchemaError(f"no feature array for utterance {utterance_id!r}")
    with np.load(path) as bundle:
        if tuple(sorted(bundle.files)) != FEATURE_ARRAYS:
            raise SchemaError(f"{path.name} holds unexpected arrays: {sorted(bundle.files)}")
        mel = bundle["mel"]
        token_map = bundle["frame_token"]
    config = FeatureConfig.from_dict(envelope["feature_config"])
    if mel.dtype != np.float32 or mel.ndim != 2 or mel.shape[1] != config.n_mels:
        raise SchemaError(
            f"{path.name}: mel must be float32 [T, {config.n_mels}], got {mel.dtype} {mel.shape}"
        )
    if token_map.dtype != np.int64 or token_map.shape != (mel.shape[0],):
        raise SchemaError(
            f"{path.name}: frame_token must be int64 [T] aligned with mel, "
            f"got {token_map.dtype} {token_map.shape}"
        )
    return mel, token_map
