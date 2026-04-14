"""Paired bootstrap confidence intervals with deterministic seeds.

Turn indices are resampled with replacement — gold and prediction travel
together (paired) — the statistic is recomputed per replicate, and the
interval is the percentile range of the replicate distribution. The seed
fully determines the result, and the replicate count is part of the
reported metadata. This quantifies sampling variability of THIS run over
THIS dataset; it makes no claim about real-world quality.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from slotvox.errors import ValidationError
from slotvox.eval.metrics import TurnRecord, micro_slot_f1, turn_accuracy
from slotvox.util.seed import derive_seed, make_rng

MIN_REPLICATES = 10
MAX_REPLICATES = 100_000


def _require_records(records) -> list[TurnRecord]:
    if isinstance(records, (str, bytes)) or not isinstance(records, (list, tuple)) or not records:
        raise ValidationError("records must be a non-empty list/tuple of TurnRecord")
    for record in records:
        if not isinstance(record, TurnRecord):
            raise ValidationError(f"records must contain TurnRecord, got {type(record).__name__}")
    return list(records)


def intent_statistic(records) -> float:
    """Intent accuracy over records."""
    items = _require_records(records)
    correct = sum(1 for item in items if item.gold_intent == item.pred_intent)
    return correct / len(items)


def slot_strict_statistic(records) -> float:
    """Micro slot F1 under the strict boundary policy."""
    return micro_slot_f1(_require_records(records), "strict").f1


def slot_partial_statistic(records) -> float:
    """Micro slot F1 under the partial-credit policy."""
    return micro_slot_f1(_require_records(records), "partial").f1


def turn_statistic(records) -> float:
    """Joint turn accuracy (strict)."""
    return turn_accuracy(_require_records(records)).accuracy


def _check_replicates(replicates: int) -> int:
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or not MIN_REPLICATES <= replicates <= MAX_REPLICATES
    ):
        raise ValidationError(
            f"replicates must be an int within [{MIN_REPLICATES}, {MAX_REPLICATES}], "
            f"got {replicates!r}"
        )
    return replicates


def _check_confidence(confidence: float) -> float:
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not np.isfinite(confidence)
        or not 0.0 < float(confidence) < 1.0
    ):
        raise ValidationError(f"confidence must be a number within (0, 1), got {confidence!r}")
    return float(confidence)


@dataclass(frozen=True)
class BootstrapCI:
    """Percentile bootstrap interval of one statistic."""

    estimate: float
    low: float
    high: float
    replicates: int
    confidence: float
    seed: int

    def __post_init__(self) -> None:
        for name in ("estimate", "low", "high"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValidationError(f"{name} must be a finite number, got {value!r}")
            object.__setattr__(self, name, float(value))
        if self.low > self.high:
            raise ValidationError(f"low {self.low} exceeds high {self.high}")
        _check_replicates(self.replicates)
        object.__setattr__(self, "confidence", _check_confidence(self.confidence))
        make_rng(self.seed)  # strict seed-range validation

    def to_dict(self) -> dict[str, Any]:
        """JSON-native dict form."""
        return {
            "estimate": self.estimate,
            "low": self.low,
            "high": self.high,
            "replicates": self.replicates,
            "confidence": self.confidence,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Any) -> BootstrapCI:
        """Reconstruct from :meth:`to_dict` output (strict)."""
        if not isinstance(data, dict):
            raise ValidationError(f"bootstrap payload must be a dict, got {type(data).__name__}")
        required = {"estimate", "low", "high", "replicates", "confidence", "seed"}
        unknown = sorted(set(data) - required)
        missing = sorted(required - set(data))
        if unknown:
            raise ValidationError(f"bootstrap payload got unknown keys: {unknown}")
        if missing:
            raise ValidationError(f"bootstrap payload is missing keys: {missing}")
        return cls(**data)


def paired_bootstrap_ci(
    records,
    statistic: Callable[..., float] = intent_statistic,
    *,
    replicates: int = 1000,
    confidence: float = 0.95,
    seed: int,
) -> BootstrapCI:
    """Percentile CI of ``statistic`` under paired index resampling."""
    items = _require_records(records)
    if not callable(statistic):
        raise ValidationError(f"statistic must be callable, got {type(statistic).__name__}")
    _check_replicates(replicates)
    level = _check_confidence(confidence)
    make_rng(seed)  # strict seed-range validation
    rng = make_rng(derive_seed("bootstrap", seed))
    count = len(items)
    estimate = float(statistic(items))
    if not np.isfinite(estimate):
        raise ValidationError("statistic returned a non-finite estimate")
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        picks = rng.integers(0, count, count)
        value = float(statistic([items[int(pick)] for pick in picks]))
        if not np.isfinite(value):
            raise ValidationError("statistic returned a non-finite replicate value")
        values[index] = value
    tail = (1.0 - level) / 2.0 * 100.0
    return BootstrapCI(
        estimate=estimate,
        low=float(np.percentile(values, tail)),
        high=float(np.percentile(values, 100.0 - tail)),
        replicates=replicates,
        confidence=level,
        seed=seed,
    )
