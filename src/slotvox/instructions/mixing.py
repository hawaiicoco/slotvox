"""Deterministic weighted mixing of instruction samples.

Mixing runs in two stages, both fully determined by ``(weights, size,
seed)``: :func:`mixing_quotas` splits ``size`` across sources with the
largest-remainder method (fractional ties broken by source name), and
:func:`mix_samples` selects per source without replacement and shuffles
the pool into the final interleaved order. No sample is ever repeated to
hit a quota — a short source is an honest error, not silent cycling.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from slotvox.errors import ValidationError
from slotvox.instructions.schema import InstructionSample
from slotvox.schema.naming import validate_id
from slotvox.util.seed import derive_seed, make_rng


def mixing_quotas(weights: Mapping[str, float], size: int) -> dict[str, int]:
    """Largest-remainder quotas over ``weights`` summing exactly to ``size``."""
    if not isinstance(weights, Mapping) or not weights:
        raise ValidationError(f"weights must be a non-empty mapping, got {type(weights).__name__}")
    total = 0.0
    for source, weight in weights.items():
        validate_id(source, "mixing source")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0.0
        ):
            raise ValidationError(
                f"mixing weight for {source!r} must be a finite number >= 0, got {weight!r}"
            )
        total += float(weight)
    if total <= 0.0:
        raise ValidationError("mixing weights must sum to a positive total")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValidationError(f"size must be a positive int, got {size!r}")
    raw = {source: size * float(weight) / total for source, weight in weights.items()}
    quotas = {source: int(math.floor(value)) for source, value in raw.items()}
    remainder = size - sum(quotas.values())
    ranked = sorted(raw, key=lambda source: (-(raw[source] - math.floor(raw[source])), source))
    for source in ranked[:remainder]:
        quotas[source] += 1
    return {source: quotas[source] for source in sorted(quotas)}


def mix_samples(
    groups: Mapping[str, Sequence[InstructionSample]],
    weights: Mapping[str, float],
    size: int,
    seed: int,
) -> tuple[InstructionSample, ...]:
    """Select and interleave samples from ``groups`` per ``weights``.

    Selection within a source and the final interleave both derive from
    ``seed`` (via :func:`derive_seed`), so a mix is exactly reproducible.
    Sources must match the weight keys in BOTH directions; groups are
    canonicalized by ``sample_id`` order before selection, so input order
    does not matter.
    """
    quotas = mixing_quotas(weights, size)
    if not isinstance(groups, Mapping):
        raise ValidationError(f"groups must be a mapping, got {type(groups).__name__}")
    unknown = sorted(set(groups) - set(quotas))
    if unknown:
        raise ValidationError(f"unknown sources in groups: {unknown}")
    missing = sorted(set(quotas) - set(groups))
    if missing:
        raise ValidationError(f"missing groups for sources: {missing}")
    make_rng(seed)  # strict seed-range validation
    pool: list[InstructionSample] = []
    seen: set[str] = set()
    for source in sorted(quotas):
        group = groups[source]
        if isinstance(group, (str, bytes)) or not isinstance(group, (list, tuple)):
            raise ValidationError(f"group {source!r} must be a list/tuple of samples")
        for sample in group:
            if not isinstance(sample, InstructionSample):
                raise ValidationError(
                    f"group {source!r} must contain InstructionSample, got {type(sample).__name__}"
                )
        quota = quotas[source]
        if quota == 0:
            continue
        ordered = sorted(group, key=lambda sample: sample.sample_id)
        if len(ordered) < quota:
            raise ValidationError(
                f"source {source!r} has {len(ordered)} samples but its quota needs {quota}"
            )
        if quota == len(ordered):
            chosen = ordered
        else:
            rng = make_rng(derive_seed("mix-select", seed, source))
            picks = sorted(
                int(index) for index in rng.choice(len(ordered), size=quota, replace=False)
            )
            chosen = [ordered[index] for index in picks]
        for sample in chosen:
            if sample.sample_id in seen:
                raise ValidationError(f"duplicate sample_id {sample.sample_id!r} across sources")
            seen.add(sample.sample_id)
        pool.extend(chosen)
    pool.sort(key=lambda sample: sample.sample_id)
    order = make_rng(derive_seed("mix-order", seed)).permutation(len(pool))
    return tuple(pool[int(index)] for index in order)
