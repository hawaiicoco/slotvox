"""Near-duplicate detection for instruction corpora (offline, deterministic).

Two complementary signals:

- exact duplicates via a canonical normalized-text hash (NFKC, casefold,
  whitespace/punctuation/control characters removed), and
- near duplicates via character 3-gram feature hashing into a
  fixed-width count vector compared by cosine similarity.

The default ``NEAR_DUPLICATE_THRESHOLD = 0.9`` is deliberately
conservative: on slotvox's synthetic corpora it flags small surface
edits of the same template (measured ~0.95 for a one-token slot change)
while distinct templates stay well below it (measured ~0.37-0.46 across
domains and languages). Both signals compare TEXT only — audio manifest
ids differ per utterance by design and are not content.
"""

from __future__ import annotations

import unicodedata

from slotvox.errors import ValidationError
from slotvox.instructions.schema import InstructionSample
from slotvox.util.jsoncanon import stable_hash

FEATURE_GRAMS = 3
FEATURE_WIDTH = 256
NEAR_DUPLICATE_THRESHOLD = 0.9


def _require_sample(sample) -> InstructionSample:
    if not isinstance(sample, InstructionSample):
        raise ValidationError(
            f"samples must contain InstructionSample, got {type(sample).__name__}"
        )
    return sample


def _require_samples(samples) -> list[InstructionSample]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, (list, tuple)):
        raise ValidationError(f"samples must be a list/tuple, got {type(samples).__name__}")
    items = [_require_sample(sample) for sample in samples]
    ids = [sample.sample_id for sample in items]
    if len(set(ids)) != len(ids):
        raise ValidationError("samples must have unique sample_ids")
    return items


def normalize_text(text: str) -> str:
    """NFKC + casefold, dropping whitespace, punctuation, and control marks."""
    if not isinstance(text, str):
        raise ValidationError(f"text must be a string, got {type(text).__name__}")
    kept = []
    for char in unicodedata.normalize("NFKC", text).casefold():
        if char.isspace():
            continue
        if unicodedata.category(char)[0] in ("P", "Z", "C"):
            continue
        kept.append(char)
    return "".join(kept)


def text_hash(text: str) -> str:
    """Canonical hash of the normalized text (the exact-duplicate signal)."""
    return stable_hash({"normalized_text": normalize_text(text)})


def sample_text(sample: InstructionSample) -> str:
    """All turn texts of ``sample`` joined by newlines (audio excluded)."""
    _require_sample(sample)
    return "\n".join(turn.text for turn in sample.turns if turn.text is not None)


def exact_duplicate_groups(samples) -> tuple[tuple[str, ...], ...]:
    """Id groups (size >= 2) sharing a normalized-text hash, sorted."""
    items = _require_samples(samples)
    buckets: dict[str, list[str]] = {}
    for sample in items:
        buckets.setdefault(text_hash(sample_text(sample)), []).append(sample.sample_id)
    groups = tuple(tuple(sorted(ids)) for ids in buckets.values() if len(ids) >= 2)
    return tuple(sorted(groups))
