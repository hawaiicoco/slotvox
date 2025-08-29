"""Leakage-free split assignment.

Two policies, both disjoint BY CONSTRUCTION:

- ``speaker``: every speaker group belongs to exactly one split, so no
  voice appears in two splits.
- ``pattern``: every pattern template belongs to exactly one split, so
  dev/test exercise phrasings the train split never saw.

Assignment is deterministic: keys are ordered by ``(derive_seed(...), key)``
so the result depends only on the keys, the counts, and the seed — never on
dict ordering or Python's hash randomization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from slotvox.config.generation import SPLITS, GenerationConfig
from slotvox.errors import ValidationError
from slotvox.util.seed import derive_seed, make_rng

SPLIT_POLICIES = ("speaker", "pattern")


def _sort_key(key: Any) -> tuple[int, Any]:
    return (0, key) if isinstance(key, int) else (1, str(key))


def assign_keys_to_splits(
    keys: Sequence[Any],
    counts: Mapping[str, int],
    seed: int,
    *,
    name: str = "key",
) -> dict[str, tuple[Any, ...]]:
    """Assign unique keys to the non-empty splits of ``counts``.

    Every non-empty split receives at least one key; keys land in exactly
    one split (disjointness by construction). Raises when there are fewer
    unique keys than non-empty splits.
    """
    if isinstance(keys, (str, bytes)) or not isinstance(keys, (list, tuple)):
        raise ValidationError(f"keys must be a list/tuple, got {type(keys).__name__}")
    if not isinstance(counts, Mapping):
        raise ValidationError(f"counts must be a mapping, got {type(counts).__name__}")
    unknown = sorted(set(counts) - set(SPLITS))
    if unknown:
        raise ValidationError(f"counts got unknown splits: {unknown}")
    for split, count in counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValidationError(f"counts[{split!r}] must be a non-negative int, got {count!r}")
    make_rng(seed)  # validates the seed
    non_empty = tuple(split for split in SPLITS if counts.get(split, 0) > 0)
    if not non_empty:
        raise ValidationError("assign_keys_to_splits needs at least one non-empty split")
    unique = list(dict.fromkeys(keys))
    if len(unique) < len(non_empty):
        raise ValidationError(
            f"need >= {len(non_empty)} unique {name}s for {len(non_empty)} non-empty "
            f"splits, got {len(unique)}"
        )
    order = sorted(unique, key=lambda key: (derive_seed("split", seed, str(key)), str(key)))
    buckets: dict[str, list[Any]] = {split: [] for split in non_empty}
    for index, key in enumerate(order):
        buckets[non_empty[index % len(non_empty)]].append(key)
    return {split: tuple(sorted(bucket, key=_sort_key)) for split, bucket in buckets.items()}


def speaker_splits(n_speakers: int, counts: Mapping[str, int], seed: int) -> dict[int, str]:
    """Map speaker ids 0..n_speakers-1 to splits (disjoint by construction)."""
    if isinstance(n_speakers, bool) or not isinstance(n_speakers, int) or n_speakers < 1:
        raise ValidationError(f"n_speakers must be a positive int, got {n_speakers!r}")
    buckets = assign_keys_to_splits(list(range(n_speakers)), counts, seed, name="speaker")
    return {speaker: split for split, speakers in buckets.items() for speaker in speakers}


def pattern_splits(
    pattern_ids: Sequence[str], counts: Mapping[str, int], seed: int
) -> dict[str, str]:
    """Map pattern ids to splits (disjoint by construction)."""
    buckets = assign_keys_to_splits(list(pattern_ids), counts, seed, name="pattern")
    return {pattern: split for split, patterns in buckets.items() for pattern in patterns}


def check_policy(policy: str) -> str:
    """Validate a split policy name."""
    if policy not in SPLIT_POLICIES:
        raise ValidationError(f"unknown split policy {policy!r}; expected one of {SPLIT_POLICIES}")
    return policy


def check_config(config: Any) -> GenerationConfig:
    """Validate that ``config`` is a GenerationConfig."""
    if not isinstance(config, GenerationConfig):
        raise ValidationError(f"config must be a GenerationConfig, got {type(config).__name__}")
    return config
