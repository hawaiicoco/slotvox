"""Intent accuracy and the tag-sequence slot F1 bridge."""

import pytest

from slotvox.errors import ValidationError
from slotvox.eval.metrics import intent_accuracy, slot_f1


def test_intent_accuracy_golden():
    score = intent_accuracy(("a", "b", "c", "d"), ("a", "b", "x", "d"))
    assert (score.correct, score.total) == (3, 4)
    assert score.accuracy == 0.75


def test_intent_validation():
    with pytest.raises(ValidationError, match="align"):
        intent_accuracy(("a",), ("a", "b"))
    with pytest.raises(ValidationError, match="at least one"):
        intent_accuracy((), ())
    with pytest.raises(ValidationError, match="non-empty"):
        intent_accuracy(("a", ""), ("a", ""))
    with pytest.raises(ValidationError, match="list/tuple"):
        intent_accuracy("ab", "ab")


def test_tag_level_slot_f1_goldens():
    gold = ("B-city", "I-city", "O", "B-day", "O")
    assert slot_f1(gold, gold).f1 == 1.0
    pred = ("B-city", "O", "O", "B-day", "O")
    assert slot_f1(gold, pred).f1 == 0.5  # city (0,2) vs (0,1) misses strictly
    assert slot_f1(gold, pred, "partial").f1 == 0.75  # credit 1.0 + 0.5 over 2
    assert slot_f1((), ()).f1 == 1.0


def test_invalid_bio_propagates():
    with pytest.raises(ValidationError):  # TaggingError is a ValidationError
        slot_f1(("O",), ("I-city",))
    with pytest.raises(ValidationError, match="align"):
        slot_f1(("O",), ("O", "O"))
