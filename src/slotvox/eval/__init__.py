"""Evaluation for joint intent-slot understanding (pure-python core).

Metrics (intent accuracy, slot F1 under strict and partial boundary
policies, joint turn accuracy), confusion matrices, metadata slice
summaries, paired bootstrap intervals, reproducible run envelopes, and
Markdown/HTML report export with mandatory limitations sections.
"""

from slotvox.eval.bootstrap import (
    BootstrapCI,
    intent_statistic,
    paired_bootstrap_ci,
    slot_partial_statistic,
    slot_strict_statistic,
    turn_statistic,
)
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
from slotvox.eval.report import html_report, markdown_report, write_reports
from slotvox.eval.runs import (
    RUN_SCHEMA_ID,
    RUN_SCHEMA_VERSION,
    ScoredRun,
    validate_run_envelope,
)
from slotvox.eval.slices import slice_summary

__all__ = [
    "BootstrapCI",
    "ConfusionMatrix",
    "IntentScore",
    "RUN_SCHEMA_ID",
    "RUN_SCHEMA_VERSION",
    "SLOT_POLICIES",
    "ScoredRun",
    "SlotScore",
    "TurnRecord",
    "TurnScore",
    "check_policy",
    "html_report",
    "intent_accuracy",
    "intent_statistic",
    "markdown_report",
    "micro_slot_f1",
    "paired_bootstrap_ci",
    "slice_pairs",
    "slice_summary",
    "slot_f1",
    "slot_partial_statistic",
    "slot_strict_statistic",
    "span_f1",
    "turn_accuracy",
    "turn_statistic",
    "validate_run_envelope",
    "write_reports",
]
