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
from collections.abc import Mapping

from slotvox.errors import ValidationError
from slotvox.schema.naming import validate_id


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
