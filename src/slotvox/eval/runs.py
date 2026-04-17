"""Reproducible evaluation runs: scored assembly and versioned envelopes.

A :class:`ScoredRun` bundles every metric computed from one list of turn
records with the provenance needed to interpret it later — run id,
metadata (model/dataset hashes belong here), and the optional bootstrap
interval. The serialized envelope is a report artifact: it summarizes
scores and slices, not the raw records (documented boundary — rerun
:meth:`ScoredRun.evaluate` on the records to reproduce it exactly).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from slotvox.errors import ValidationError
from slotvox.eval.bootstrap import BootstrapCI, paired_bootstrap_ci
from slotvox.eval.confusion import ConfusionMatrix
from slotvox.eval.metrics import (
    IntentScore,
    SlotScore,
    TurnRecord,
    TurnScore,
    intent_accuracy,
    micro_slot_f1,
    turn_accuracy,
)
from slotvox.eval.slices import slice_summary
from slotvox.schema.naming import validate_id
from slotvox.util.jsoncanon import stable_hash

RUN_SCHEMA_ID = "slotvox.eval-run"
RUN_SCHEMA_VERSION = 1


def _check_metadata(metadata) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValidationError(f"metadata must be a dict, got {type(metadata).__name__}")
    try:
        stable_hash(metadata)
    except ValidationError as exc:
        raise ValidationError(f"metadata must be JSON-native: {exc}") from exc
    return dict(metadata)


def _require_records(records) -> tuple[TurnRecord, ...]:
    if isinstance(records, (str, bytes)) or not isinstance(records, (list, tuple)) or not records:
        raise ValidationError("records must be a non-empty list/tuple of TurnRecord")
    for record in records:
        if not isinstance(record, TurnRecord):
            raise ValidationError(f"records must contain TurnRecord, got {type(record).__name__}")
    return tuple(records)


@dataclass(frozen=True)
class ScoredRun:
    """Every score for one evaluated record list, with provenance."""

    run_id: str
    records: tuple[TurnRecord, ...]
    intent: IntentScore
    slot_strict: SlotScore
    slot_partial: SlotScore
    turn: TurnScore
    confusion: ConfusionMatrix
    slices: dict[str, dict[str, dict[str, Any]]]
    metadata: dict[str, Any]
    bootstrap: BootstrapCI | None = None

    def __post_init__(self) -> None:
        validate_id(self.run_id, "run_id")
        records = self.records
        if isinstance(records, list):
            records = tuple(records)
            object.__setattr__(self, "records", records)
        _require_records(records)
        if not isinstance(self.intent, IntentScore):
            raise ValidationError("intent must be an IntentScore")
        for name in ("slot_strict", "slot_partial"):
            if not isinstance(getattr(self, name), SlotScore):
                raise ValidationError(f"{name} must be a SlotScore")
        if not isinstance(self.turn, TurnScore):
            raise ValidationError("turn must be a TurnScore")
        if not isinstance(self.confusion, ConfusionMatrix):
            raise ValidationError("confusion must be a ConfusionMatrix")
        if not isinstance(self.slices, dict):
            raise ValidationError("slices must be a dict")
        if not isinstance(self.metadata, dict):
            raise ValidationError("metadata must be a dict")
        if self.bootstrap is not None and not isinstance(self.bootstrap, BootstrapCI):
            raise ValidationError("bootstrap must be a BootstrapCI or None")

    @property
    def record_count(self) -> int:
        """Number of evaluated turns."""
        return len(self.records)

    @classmethod
    def evaluate(
        cls,
        run_id: str,
        records,
        *,
        metadata: dict[str, Any] | None = None,
        bootstrap_seed: int | None = None,
        replicates: int = 1000,
        confidence: float = 0.95,
    ) -> ScoredRun:
        """Compute every metric from ``records`` in one reproducible pass."""
        validate_id(run_id, "run_id")
        items = _require_records(records)
        gold = [record.gold_intent for record in items]
        pred = [record.pred_intent for record in items]
        bootstrap = None
        if bootstrap_seed is not None:
            bootstrap = paired_bootstrap_ci(
                items, replicates=replicates, confidence=confidence, seed=bootstrap_seed
            )
        return cls(
            run_id=run_id,
            records=items,
            intent=intent_accuracy(gold, pred),
            slot_strict=micro_slot_f1(items, "strict"),
            slot_partial=micro_slot_f1(items, "partial"),
            turn=turn_accuracy(items),
            confusion=ConfusionMatrix.from_pairs(gold, pred),
            slices=slice_summary(items),
            metadata={} if metadata is None else _check_metadata(metadata),
            bootstrap=bootstrap,
        )
