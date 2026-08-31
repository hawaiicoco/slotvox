"""Encoded examples from saved datasets + feature stores.

The numpy side stays torch-free: :func:`encode_split` turns manifest rows
and stored features into per-frame label indices. It validates that the
feature store belongs to the dataset (hash match), that the stored frame
alignment agrees with the manifest, and that every intent/tag exists in
the given vocabularies — unknown labels are rejected, never skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from slotvox.config import FeatureConfig
from slotvox.config.base import config_from_dict
from slotvox.config.generation import SPLITS
from slotvox.data.alignment import frame_span_tags, frame_token_map
from slotvox.data.features_store import load_features, read_features_envelope
from slotvox.data.persist import DATASET_DIR_SCHEMA_ID
from slotvox.errors import SchemaError, SlotvoxError, ValidationError
from slotvox.models.vocab import PAD_INDEX, Vocab
from slotvox.schema.annotations import AnnotatedUtterance
from slotvox.schema.serialize import read_json, read_jsonl
from slotvox.synth.utterance import SyntheticUtterance

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class EncodedExample:
    """One example as model-ready arrays plus its label names."""

    utterance_id: str
    mel: np.ndarray  # [T, F] float32
    frame_tag_indices: np.ndarray  # [T] int64, vocab indices per frame
    intent_index: int
    frame_tag_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.utterance_id, str) or not self.utterance_id:
            raise ValidationError("EncodedExample.utterance_id must be a non-empty string")
        if not isinstance(self.mel, np.ndarray) or self.mel.ndim != 2:
            raise ValidationError("EncodedExample.mel must be a 2-D numpy array")
        if self.mel.dtype != np.float32:
            raise ValidationError(f"EncodedExample.mel must be float32, got {self.mel.dtype}")
        if not isinstance(self.frame_tag_indices, np.ndarray):
            raise ValidationError("EncodedExample.frame_tag_indices must be a numpy array")
        if self.frame_tag_indices.dtype != np.int64 or self.frame_tag_indices.ndim != 1:
            raise ValidationError("frame_tag_indices must be 1-D int64")
        if self.frame_tag_indices.shape != (self.mel.shape[0],):
            raise ValidationError(
                f"frame_tag_indices length {self.frame_tag_indices.shape[0]} != "
                f"mel frames {self.mel.shape[0]}"
            )
        if self.frame_tag_indices.size and int(self.frame_tag_indices.min()) < 0:
            raise ValidationError("frame_tag_indices must be non-negative vocab indices")
        if isinstance(self.intent_index, bool) or not isinstance(self.intent_index, int):
            raise ValidationError(f"intent_index must be an int, got {self.intent_index!r}")
        if self.intent_index < 0:
            raise ValidationError(f"intent_index must be >= 0, got {self.intent_index}")
        names = self.frame_tag_names
        if isinstance(names, list):
            object.__setattr__(self, "frame_tag_names", tuple(names))
            names = self.frame_tag_names
        if not isinstance(names, tuple) or len(names) != self.mel.shape[0]:
            raise ValidationError("frame_tag_names must be a tuple with one name per frame")
        if not all(isinstance(name, str) and name for name in names):
            raise ValidationError("frame_tag_names must contain non-empty strings")

    @property
    def n_frames(self) -> int:
        """Frame count T."""
        return int(self.mel.shape[0])


def encode_split(
    dataset_dir: str | Path,
    features_dir: str | Path,
    split: str,
    *,
    intents: Vocab,
    tags: Vocab,
) -> list[EncodedExample]:
    """Encode one split of a saved dataset into :class:`EncodedExample`s."""
    if split not in SPLITS:
        raise ValidationError(f"split must be one of {SPLITS}, got {split!r}")
    for name, vocab in (("intents", intents), ("tags", tags)):
        if not isinstance(vocab, Vocab):
            raise ValidationError(f"{name} must be a Vocab, got {type(vocab).__name__}")
    root = Path(dataset_dir)
    envelope = read_json(root / "dataset.json")
    if not isinstance(envelope, dict) or envelope.get("schema") != DATASET_DIR_SCHEMA_ID:
        raise SchemaError(f"{root} does not hold a slotvox.dataset directory")
    feature_envelope = read_features_envelope(features_dir)
    if feature_envelope.get("dataset_hash") != envelope.get("dataset_hash"):
        raise SchemaError("feature store does not belong to this dataset (hash mismatch)")
    feature_config = config_from_dict(feature_envelope["feature_config"])
    if not isinstance(feature_config, FeatureConfig):
        raise SchemaError("feature envelope does not describe a FeatureConfig")
    rows = read_jsonl(root / "manifest.jsonl")
    encoded: list[EncodedExample] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SchemaError(f"manifest row {index}: expected an object")
        if row.get("split") != split:
            continue
        encoded.append(_encode_row(row, Path(features_dir), feature_config, intents, tags, index))
    return encoded


def _encode_row(
    row: dict,
    features_dir: Path,
    feature_config: FeatureConfig,
    intents: Vocab,
    tags: Vocab,
    index: int,
) -> EncodedExample:
    annotation = AnnotatedUtterance.from_dict(row["annotation"])
    n_samples = row.get("n_samples")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise SchemaError(f"manifest row {index}: invalid n_samples {n_samples!r}")
    utterance = SyntheticUtterance.from_dict(
        {
            "sample_rate": row.get("sample_rate"),
            "n_samples": n_samples,
            "snr_db": row.get("snr_db"),
            "segments": row.get("segments"),
        },
        np.zeros(n_samples, dtype=np.float32),
    )
    token_map = frame_token_map(utterance, feature_config)
    names = frame_span_tags(annotation.tags, token_map)
    mel, stored_map = load_features(features_dir, annotation.utterance_id)
    if not np.array_equal(stored_map, token_map):
        raise SchemaError(
            f"frame alignment for {annotation.utterance_id!r} disagrees with the manifest"
        )
    if mel.shape[0] != len(names):
        raise SchemaError(
            f"feature frames {mel.shape[0]} != aligned tags {len(names)} "
            f"for {annotation.utterance_id!r}"
        )
    try:
        intent_index = intents.index(annotation.intent)
        tag_indices = tags.indices(names)
    except ValidationError as exc:
        raise SchemaError(f"manifest row {index} ({annotation.utterance_id}): {exc}") from exc
    return EncodedExample(
        utterance_id=annotation.utterance_id,
        mel=mel,
        frame_tag_indices=np.asarray(tag_indices, dtype=np.int64),
        intent_index=intent_index,
        frame_tag_names=names,
    )


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SlotvoxError(
            "batching requires the torch extra: pip install 'slotvox[torch]' "
            "(CPU wheels: --index-url https://download.pytorch.org/whl/cpu)"
        ) from exc
    return torch


@dataclass(frozen=True)
class FrameBatch:
    """A padded batch of encoded examples (torch tensors)."""

    mel: torch.Tensor  # [B, T, F] float32, zero-padded  # noqa: F821
    mask: torch.Tensor  # [B, T] bool, True = real frame  # noqa: F821
    intent: torch.Tensor  # [B] int64  # noqa: F821
    tags: torch.Tensor  # [B, T] int64, PAD_INDEX where masked  # noqa: F821

    @property
    def size(self) -> int:
        """Batch size B."""
        return int(self.intent.shape[0])


def collate(examples) -> FrameBatch:
    """Pad a non-empty sequence of examples into one batch."""
    if (
        isinstance(examples, (str, bytes))
        or not isinstance(examples, (list, tuple))
        or not examples
    ):
        raise ValidationError("collate needs a non-empty list of EncodedExample")
    for example in examples:
        if not isinstance(example, EncodedExample):
            raise ValidationError(
                f"collate needs EncodedExample instances, got {type(example).__name__}"
            )
    width = max(example.n_frames for example in examples)
    features = examples[0].mel.shape[1]
    mel = np.zeros((len(examples), width, features), dtype=np.float32)
    mask = np.zeros((len(examples), width), dtype=bool)
    intent = np.zeros(len(examples), dtype=np.int64)
    tags = np.full((len(examples), width), PAD_INDEX, dtype=np.int64)
    for row, example in enumerate(examples):
        if example.mel.shape[1] != features:
            raise ValidationError(
                f"examples disagree on feature width: {example.mel.shape[1]} vs {features}"
            )
        length = example.n_frames
        mel[row, :length] = example.mel
        mask[row, :length] = True
        intent[row] = example.intent_index
        tags[row, :length] = example.frame_tag_indices
    torch = _torch()
    return FrameBatch(
        mel=torch.from_numpy(mel),
        mask=torch.from_numpy(mask),
        intent=torch.from_numpy(intent),
        tags=torch.from_numpy(tags),
    )


def iterate_batches(examples, batch_size: int, *, rng=None):
    """Yield batches; sequential when ``rng`` is None, shuffled when given."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValidationError(f"batch_size must be a positive int, got {batch_size!r}")
    items = list(examples)
    if not items:
        raise ValidationError("iterate_batches needs at least one example")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise ValidationError("rng must be a numpy Generator (see make_rng) or None")
    order = np.arange(len(items)) if rng is None else rng.permutation(len(items))
    for start in range(0, len(items), batch_size):
        chunk = [items[int(i)] for i in order[start : start + batch_size]]
        yield collate(chunk)
