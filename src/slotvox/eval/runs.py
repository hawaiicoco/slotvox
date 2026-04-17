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

from slotvox.errors import SchemaError, ValidationError
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

    def to_dict(self) -> dict[str, Any]:
        """The serialized report envelope (records themselves excluded)."""
        return {
            "schema": RUN_SCHEMA_ID,
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": self.run_id,
            "record_count": self.record_count,
            "intent": {"correct": self.intent.correct, "total": self.intent.total},
            "slot_strict": _slot_dict(self.slot_strict),
            "slot_partial": _slot_dict(self.slot_partial),
            "turn": {"correct": self.turn.correct, "total": self.turn.total},
            "confusion": self.confusion.to_dict(),
            "slices": self.slices,
            "metadata": self.metadata,
            "bootstrap": None if self.bootstrap is None else self.bootstrap.to_dict(),
        }

    @property
    def run_hash(self) -> str:
        """Provenance hash of the envelope."""
        return stable_hash(self.to_dict())

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


_SCORE_KEYS = frozenset({"gold_count", "pred_count", "credit", "precision", "recall", "f1"})
_COUNT_KEYS = frozenset({"correct", "total"})
_SLICE_KEYS = frozenset({"count", "intent_accuracy", "slot_f1_strict", "slot_f1_partial"})
RUN_ENVELOPE_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "run_id",
        "record_count",
        "intent",
        "slot_strict",
        "slot_partial",
        "turn",
        "confusion",
        "slices",
        "metadata",
        "bootstrap",
    }
)


def _slot_dict(score: SlotScore) -> dict[str, Any]:
    return {
        "gold_count": score.gold_count,
        "pred_count": score.pred_count,
        "credit": score.credit,
        "precision": score.precision,
        "recall": score.recall,
        "f1": score.f1,
    }


def _check_counts(payload: Any, label: str) -> None:
    if not isinstance(payload, dict) or set(payload) != _COUNT_KEYS:
        raise SchemaError(f"{label} must be a dict with keys {sorted(_COUNT_KEYS)}")
    correct, total = payload["correct"], payload["total"]
    for name, value in (("correct", correct), ("total", total)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SchemaError(f"{label} {name} must be a non-negative int")
    if correct > total:
        raise SchemaError(f"{label} correct {correct} exceeds total {total}")


def _check_slot_payload(payload: Any, label: str) -> None:
    if not isinstance(payload, dict) or set(payload) != _SCORE_KEYS:
        raise SchemaError(f"{label} must be a dict with keys {sorted(_SCORE_KEYS)}")
    try:
        SlotScore(
            gold_count=payload["gold_count"],
            pred_count=payload["pred_count"],
            credit=payload["credit"],
        )
    except ValidationError as exc:
        raise SchemaError(f"{label} is invalid: {exc}") from exc


def _envelope_error(payload: Any, where: str, problem: str) -> None:
    raise SchemaError(f"run envelope {where} is invalid: {problem}")


def validate_run_envelope(payload: Any) -> dict[str, Any]:
    """Strict envelope validation; returns the payload when valid."""
    if not isinstance(payload, dict):
        raise SchemaError(f"run envelope must be a dict, got {type(payload).__name__}")
    unknown = sorted(set(payload) - RUN_ENVELOPE_KEYS)
    missing = sorted(RUN_ENVELOPE_KEYS - set(payload))
    if unknown:
        raise SchemaError(f"run envelope got unknown keys: {unknown}")
    if missing:
        raise SchemaError(f"run envelope is missing keys: {missing}")
    if payload["schema"] != RUN_SCHEMA_ID:
        raise SchemaError(f"unknown eval-run schema {payload['schema']!r}")
    if payload["schema_version"] != RUN_SCHEMA_VERSION:
        raise SchemaError(f"unsupported eval-run schema version {payload['schema_version']!r}")
    if not isinstance(payload["run_id"], str) or not payload["run_id"]:
        raise SchemaError("run envelope run_id must be a non-empty string")
    count = payload["record_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise SchemaError("run envelope record_count must be a positive int")
    _check_counts(payload["intent"], "intent score")
    _check_counts(payload["turn"], "turn score")
    _check_slot_payload(payload["slot_strict"], "slot_strict")
    _check_slot_payload(payload["slot_partial"], "slot_partial")
    try:
        ConfusionMatrix.from_dict(payload["confusion"])
    except ValidationError as exc:
        raise SchemaError(f"run envelope confusion is invalid: {exc}") from exc
    slices = payload["slices"]
    if not isinstance(slices, dict):
        raise SchemaError("run envelope slices must be a dict")
    for kind, values in slices.items():
        if not isinstance(kind, str) or not isinstance(values, dict):
            _envelope_error(payload, "slices", f"kind {kind!r} must map to a dict")
        for value, entry in values.items():
            if not isinstance(value, str) or not isinstance(entry, dict):
                _envelope_error(payload, "slices", f"value {value!r} must map to a dict")
            if set(entry) != _SLICE_KEYS:
                _envelope_error(payload, "slices", f"entry keys {sorted(set(entry))}")
    if not isinstance(payload["metadata"], dict):
        raise SchemaError("run envelope metadata must be a dict")
    bootstrap = payload["bootstrap"]
    if bootstrap is not None:
        try:
            BootstrapCI.from_dict(bootstrap)
        except ValidationError as exc:
            raise SchemaError(f"run envelope bootstrap is invalid: {exc}") from exc
    return payload
