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

import numpy as np

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


def feature_vector(text: str) -> np.ndarray:
    """L2-normalized character 3-gram count vector of width ``FEATURE_WIDTH``.

    Text shorter than one gram hashes as a whole; text that normalizes to
    nothing yields the zero vector (similarity 0 against everything).
    """
    normalized = normalize_text(text)
    vector = np.zeros(FEATURE_WIDTH, dtype=np.float64)
    if not normalized:
        return vector
    if len(normalized) < FEATURE_GRAMS:
        pieces = [normalized]
    else:
        last = len(normalized) - FEATURE_GRAMS + 1
        pieces = [normalized[start : start + FEATURE_GRAMS] for start in range(last)]
    for piece in pieces:
        vector[int(stable_hash(piece)[:8], 16) % FEATURE_WIDTH] += 1.0
    magnitude = float(np.linalg.norm(vector))
    if magnitude == 0.0:
        return vector
    return vector / magnitude


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine of two equal-length 1-D vectors; a zero vector gives 0.0."""
    for name, array in (("left", left), ("right", right)):
        if not isinstance(array, np.ndarray) or array.ndim != 1:
            raise ValidationError(f"{name} must be a 1-D numpy array, got {type(array).__name__}")
    if left.shape != right.shape:
        raise ValidationError(f"vector shape mismatch: {left.shape} vs {right.shape}")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def near_duplicate_pairs(
    samples, *, threshold: float = NEAR_DUPLICATE_THRESHOLD
) -> tuple[tuple[str, str, float], ...]:
    """Sample-id pairs whose text vectors cosine-match at >= ``threshold``.

    Pairs are ordered by sample id and the result is fully deterministic.
    Exact duplicates appear here too (similarity 1.0); consumers that want
    only the near misses subtract :func:`exact_duplicate_groups`.
    """
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ValidationError(f"threshold must be a number within [0, 1], got {threshold!r}")
    items = _require_samples(samples)
    ordered = sorted(items, key=lambda sample: sample.sample_id)
    vectors = [feature_vector(sample_text(sample)) for sample in ordered]
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            similarity = cosine_similarity(vectors[i], vectors[j])
            if similarity >= float(threshold):
                pairs.append((ordered[i].sample_id, ordered[j].sample_id, similarity))
    return tuple(pairs)
