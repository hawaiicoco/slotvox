"""Golden pin for the eval package API surface."""

import slotvox.eval as evaluation

GOLDEN_ALL = [
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


def test_api_surface_is_pinned():
    assert sorted(evaluation.__all__) == GOLDEN_ALL
    for name in GOLDEN_ALL:
        assert hasattr(evaluation, name)


def test_schema_constants():
    assert evaluation.RUN_SCHEMA_ID == "slotvox.eval-run"
    assert evaluation.RUN_SCHEMA_VERSION == 1
    assert evaluation.SLOT_POLICIES == ("strict", "partial")
