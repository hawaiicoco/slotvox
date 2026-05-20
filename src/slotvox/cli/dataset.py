"""``slotvox generate`` / ``slotvox featurize`` — synthetic data pipelines."""

from __future__ import annotations

from slotvox.config.generation import SPLITS
from slotvox.errors import ValidationError


def parse_counts(text: str) -> dict[str, int]:
    """Parse ``"train=8,dev=2"`` strictly into a counts dict."""
    if not isinstance(text, str) or not text:
        raise ValidationError("counts must be a non-empty string like 'train=8,dev=2'")
    counts: dict[str, int] = {}
    for part in text.split(","):
        if "=" not in part:
            raise ValidationError(f"count entry {part!r} must look like 'split=N'")
        key, _, raw = part.partition("=")
        if key not in SPLITS:
            raise ValidationError(f"unknown split {key!r}; expected one of {SPLITS}")
        if key in counts:
            raise ValidationError(f"duplicate split {key!r} in counts")
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValidationError(f"count for {key!r} must be an int, got {raw!r}") from exc
        if value < 0:
            raise ValidationError(f"count for {key!r} must be >= 0, got {value}")
        counts[key] = value
    return counts
