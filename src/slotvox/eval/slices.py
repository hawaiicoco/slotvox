"""Slice summaries: metrics broken down by turn metadata.

Slices answer questions like "does slot F1 hold up at 15 dB too?". Each
record's ``slices`` pairs (kind -> value, e.g. ``noise -> snr-15``)
define its groups; every group reports count, intent accuracy, and both
slot policies side by side. Output dicts are ordered by kind then value,
so summaries are deterministic and diffable.
"""

from __future__ import annotations

from typing import Any

from slotvox.errors import ValidationError
from slotvox.eval.metrics import TurnRecord, micro_slot_f1


def slice_summary(records) -> dict[str, dict[str, dict[str, Any]]]:
    """Group ``records`` by their slice pairs and score every group."""
    if isinstance(records, (str, bytes)) or not isinstance(records, (list, tuple)):
        raise ValidationError(f"records must be a list/tuple, got {type(records).__name__}")
    for record in records:
        if not isinstance(record, TurnRecord):
            raise ValidationError(f"records must contain TurnRecord, got {type(record).__name__}")
    groups: dict[tuple[str, str], list[TurnRecord]] = {}
    for record in records:
        for pair in record.slices:
            groups.setdefault(pair, []).append(record)
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for kind, value in sorted(groups):
        group = groups[(kind, value)]
        correct = sum(1 for item in group if item.gold_intent == item.pred_intent)
        summary.setdefault(kind, {})[value] = {
            "count": len(group),
            "intent_accuracy": correct / len(group),
            "slot_f1_strict": micro_slot_f1(group, "strict").f1,
            "slot_f1_partial": micro_slot_f1(group, "partial").f1,
        }
    return summary
