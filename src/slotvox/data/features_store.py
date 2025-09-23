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

import numpy as np

from slotvox.config.features import FeatureConfig
from slotvox.data.alignment import frame_token_map
from slotvox.data.dataset import GeneratedDataset
from slotvox.errors import ValidationError
from slotvox.features.frontend import log_mel
from slotvox.schema.serialize import write_json_atomic

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
