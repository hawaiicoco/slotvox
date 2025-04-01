"""Deterministic seeding helpers.

All randomness in slotvox flows through :func:`make_rng`, so a given seed
always produces the same stream, and through :func:`derive_seed`, so named
experiment components (splits, noise draws, bootstrap resamples) get
independent-but-reproducible seeds without manual bookkeeping.
"""

from __future__ import annotations

import numpy as np

from slotvox.errors import ValidationError
from slotvox.util.jsoncanon import stable_hash

SEED_MAX = 2**32 - 1


def make_rng(seed: int) -> np.random.Generator:
    """Return a NumPy PCG64 generator seeded with ``seed`` (0..SEED_MAX)."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValidationError(f"seed must be an int, got {type(seed).__name__}")
    if not 0 <= seed <= SEED_MAX:
        raise ValidationError(f"seed must be within [0, {SEED_MAX}], got {seed}")
    return np.random.Generator(np.random.PCG64(seed))


def derive_seed(*parts: object) -> int:
    """Derive a deterministic seed in [0, SEED_MAX] from JSON-native parts.

    Different part sequences yield different seeds with overwhelming
    probability (SHA-256 truncated to 32 bits of seed space); identical
    parts always yield the identical seed.
    """
    if not parts:
        raise ValidationError("derive_seed requires at least one part")
    digest = stable_hash(list(parts))
    return int(digest[:16], 16) % (SEED_MAX + 1)
