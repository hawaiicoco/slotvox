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
from slotvox.data.acoustics import token_segment_plan
from slotvox.data.lexicon import slot_entries
from slotvox.data.patterns import PatternTemplate
from slotvox.errors import ValidationError
from slotvox.schema.annotations import AnnotatedUtterance
from slotvox.schema.domains import DomainSpec
from slotvox.synth.utterance import SyntheticUtterance, render_utterance
from slotvox.tagging.bio import OUTSIDE, format_tag, parse_tag
from slotvox.tagging.tokenize import tokenize
from slotvox.util.seed import make_rng


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


def instantiate_template(
    domain_id: str,
    language: str,
    template: PatternTemplate,
    rng: np.random.Generator,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Fill a template: tokens, BIO tags, and per-slot surface values."""
    if not isinstance(domain_id, str) or not domain_id:
        raise ValidationError("domain_id must be a non-empty string")
    if not isinstance(template, PatternTemplate):
        raise ValidationError(f"template must be a PatternTemplate, got {type(template).__name__}")
    if not isinstance(rng, np.random.Generator):
        raise ValidationError(
            f"rng must be a numpy Generator (see make_rng), got {type(rng).__name__}"
        )
    tokens: list[str] = []
    tags: list[str] = []
    slot_values: dict[str, list[str]] = {}
    for plan in template.tokens:
        if not plan.is_slot:
            tokens.append(plan.text)
            tags.append(OUTSIDE)
            continue
        entries = slot_entries(domain_id, language, plan.slot)
        entry = entries[int(rng.integers(len(entries)))]
        for index, token in enumerate(tokenize(entry.surface, language)):
            tokens.append(token)
            tags.append(format_tag("B" if index == 0 else "I", plan.slot))
        slot_values.setdefault(plan.slot, []).append(entry.surface)
    return tuple(tokens), tuple(tags), {slot: tuple(v) for slot, v in slot_values.items()}


def render_example(
    domain: DomainSpec,
    template: PatternTemplate,
    language: str,
    *,
    utterance_id: str,
    split: str,
    speaker_id: int,
    snr_db: float | None,
    seed: int,
    sample_rate: int = 16000,
    token_ms: int = 80,
    gap_ms: int = 20,
) -> GeneratedExample:
    """Render one fully labeled example; deterministic given ``seed``."""
    if not isinstance(domain, DomainSpec):
        raise ValidationError(f"domain must be a DomainSpec, got {type(domain).__name__}")
    rng = make_rng(seed)
    tokens, tags, slot_values = instantiate_template(domain.domain, language, template, rng)
    annotation = AnnotatedUtterance(
        utterance_id=utterance_id,
        language=language,
        tokens=tokens,
        tags=tags,
        intent=template.intent,
    )
    annotation.validate_against(domain)
    plans = []
    for token, tag in zip(tokens, tags, strict=True):
        prefix, slot = parse_tag(tag)
        plans.append(
            token_segment_plan(
                domain.domain,
                token,
                None if prefix == OUTSIDE else slot,
                duration_ms=token_ms,
                speaker_id=speaker_id,
            )
        )
    utterance = render_utterance(plans, sample_rate, gap_ms=gap_ms, snr_db=snr_db, seed=seed)
    return GeneratedExample(
        utterance_id=utterance_id,
        split=split,
        speaker_id=speaker_id,
        snr_db=snr_db,
        pattern_id=template.pattern_id,
        annotation=annotation,
        utterance=utterance,
        slot_values=slot_values,
    )
