"""Evaluation for joint intent-slot understanding (pure-python core).

Metrics (intent accuracy, slot F1 under strict and partial boundary
policies, joint turn accuracy), confusion matrices, and metadata slice
summaries. Reproducible run metadata and report export build on top of
these primitives.
"""

from slotvox.eval.confusion import ConfusionMatrix
from slotvox.eval.metrics import (
    SLOT_POLICIES,
    IntentScore,
    SlotScore,
    TurnRecord,
    TurnScore,
    check_policy,
    intent_accuracy,
    micro_slot_f1,
    slice_pairs,
    slot_f1,
    span_f1,
    turn_accuracy,
)
from slotvox.eval.slices import slice_summary

__all__ = [
    "ConfusionMatrix",
    "IntentScore",
    "SLOT_POLICIES",
    "SlotScore",
    "TurnRecord",
    "TurnScore",
    "check_policy",
    "intent_accuracy",
    "micro_slot_f1",
    "slice_pairs",
    "slice_summary",
    "slot_f1",
    "span_f1",
    "turn_accuracy",
]
