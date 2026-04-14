"""Golden pin for the eval package API surface."""

import slotvox.eval as evaluation

GOLDEN_ALL = [
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


def test_api_surface_is_pinned():
    assert sorted(evaluation.__all__) == GOLDEN_ALL
    for name in GOLDEN_ALL:
        assert hasattr(evaluation, name)


def test_policy_constant():
    assert evaluation.SLOT_POLICIES == ("strict", "partial")
