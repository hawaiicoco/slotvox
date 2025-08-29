"""Synthetic dialogue factory: templates -> rendered, labeled utterances.

Turns pattern templates and lexicons into utterance waveforms with token-
level BIO annotations aligned to signal segments BY CONSTRUCTION. NOT real
speech: every clip is deterministic synthetic audio, and every label was
placed by the generator — never by an annotator or a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from slotvox.config.generation import SPLITS
from slotvox.errors import ValidationError
from slotvox.schema.annotations import AnnotatedUtterance
from slotvox.synth.utterance import SyntheticUtterance
from slotvox.tagging.bio import format_tag


@dataclass(frozen=True)
class GeneratedExample:
    """One generated utterance with its by-construction annotation."""

    utterance_id: str
    split: str
    speaker_id: int
    snr_db: float | None
    pattern_id: str
    annotation: AnnotatedUtterance
    utterance: SyntheticUtterance
    slot_values: dict[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if not isinstance(self.utterance_id, str) or not self.utterance_id:
            raise ValidationError("GeneratedExample.utterance_id must be a non-empty string")
        if any(ch.isspace() for ch in self.utterance_id):
            raise ValidationError("GeneratedExample.utterance_id must not contain whitespace")
        if self.split not in SPLITS:
            raise ValidationError(f"split must be one of {SPLITS}, got {self.split!r}")
        if (
            isinstance(self.speaker_id, bool)
            or not isinstance(self.speaker_id, int)
            or self.speaker_id < 0
        ):
            raise ValidationError(f"speaker_id must be a non-negative int, got {self.speaker_id!r}")
        if self.snr_db is not None:
            if (
                isinstance(self.snr_db, bool)
                or not isinstance(self.snr_db, (int, float))
                or not np.isfinite(self.snr_db)
            ):
                raise ValidationError(f"snr_db must be finite or None, got {self.snr_db!r}")
            object.__setattr__(self, "snr_db", float(self.snr_db))
        if not isinstance(self.pattern_id, str) or not self.pattern_id:
            raise ValidationError("GeneratedExample.pattern_id must be a non-empty string")
        if not isinstance(self.annotation, AnnotatedUtterance):
            raise ValidationError("GeneratedExample.annotation must be an AnnotatedUtterance")
        if not isinstance(self.utterance, SyntheticUtterance):
            raise ValidationError("GeneratedExample.utterance must be a SyntheticUtterance")
        if not isinstance(self.slot_values, dict):
            raise ValidationError("GeneratedExample.slot_values must be a dict")
        normalized: dict[str, tuple[str, ...]] = {}
        for slot, surfaces in self.slot_values.items():
            items = tuple(surfaces) if isinstance(surfaces, list) else surfaces
            if (
                not isinstance(items, tuple)
                or not items
                or not all(isinstance(surface, str) and surface for surface in items)
            ):
                raise ValidationError(f"slot_values[{slot!r}] must be a non-empty tuple of strings")
            normalized[slot] = items
        object.__setattr__(self, "slot_values", normalized)
        if self.annotation.utterance_id != self.utterance_id:
            raise ValidationError("annotation.utterance_id does not match example id")
        if self.utterance.snr_db != self.snr_db:
            raise ValidationError("utterance.snr_db does not match example snr_db")
        if len(self.annotation.tokens) != len(self.utterance.segments):
            raise ValidationError(
                f"annotation/utterance misaligned: {len(self.annotation.tokens)} tokens "
                f"vs {len(self.utterance.segments)} segments"
            )
        if set(normalized) != set(self.annotation.tag_slot_names()):
            raise ValidationError("slot_values keys must equal the tagged slot names")
        for slot, surfaces in normalized.items():
            starts = sum(1 for tag in self.annotation.tags if tag == format_tag("B", slot))
            if starts != len(surfaces):
                raise ValidationError(
                    f"slot {slot!r}: {len(surfaces)} values vs {starts} tagged spans"
                )

    def to_dict(self) -> dict[str, Any]:
        """Manifest row: everything needed to pair audio with labels."""
        return {
            "utterance_id": self.utterance_id,
            "split": self.split,
            "speaker_id": self.speaker_id,
            "snr_db": self.snr_db,
            "pattern_id": self.pattern_id,
            "annotation": self.annotation.to_dict(),
            "slot_values": {slot: list(surfaces) for slot, surfaces in self.slot_values.items()},
            "sample_rate": self.utterance.sample_rate,
            "n_samples": self.utterance.n_samples,
            "segments": [segment.to_dict() for segment in self.utterance.segments],
        }
