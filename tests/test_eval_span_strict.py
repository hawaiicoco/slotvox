"""Strict span scoring goldens (all values hand-computed)."""

import pytest

from slotvox.errors import ValidationError
from slotvox.eval.metrics import SlotScore, span_f1
from slotvox.tagging.bio import Span

GOLD = (("city", 0, 2), ("day", 3, 4))


def test_perfect_and_boundary_miss():
    score = span_f1(GOLD, GOLD)
    assert (score.gold_count, score.pred_count, score.credit) == (2, 2, 2.0)
    assert (score.precision, score.recall, score.f1) == (1.0, 1.0, 1.0)
    miss = span_f1(GOLD, (("city", 0, 2), ("day", 3, 5)))
    assert miss.credit == 1.0
    assert (miss.precision, miss.recall, miss.f1) == (0.5, 0.5, 0.5)


def test_label_mismatch_and_multiplicity():
    assert span_f1(GOLD, (("city", 0, 2), ("time", 3, 4))).credit == 1.0
    doubled = (("city", 0, 1), ("city", 0, 1))
    assert span_f1(doubled, (("city", 0, 1),)).credit == 1.0
    assert span_f1(doubled, (("city", 0, 1),) * 2).credit == 2.0


def test_empty_side_conventions():
    empty = span_f1((), ())
    assert (empty.gold_count, empty.pred_count, empty.credit) == (0, 0, 0.0)
    assert empty.f1 == 1.0
    spurious = span_f1((), (("city", 0, 1),))
    assert spurious.precision == 0.0
    assert spurious.recall == 1.0
    assert spurious.f1 == 0.0


def test_span_objects_and_tuples_equivalent():
    assert span_f1(GOLD, (Span("city", 0, 2), Span("day", 3, 4))).credit == 2.0


def test_input_validation():
    with pytest.raises(ValidationError, match="list/tuple"):
        span_f1(GOLD, "nope")
    with pytest.raises(ValidationError, match="Span"):
        span_f1(GOLD, ((1, 2, 3, 4),))
    with pytest.raises(ValidationError):
        span_f1(GOLD, ((1, 0, 2),))  # non-string label
    with pytest.raises(ValidationError, match="credit"):
        SlotScore(1, 1, 5.0)
    with pytest.raises(ValidationError, match="non-negative"):
        SlotScore(-1, 0, 0.0)
