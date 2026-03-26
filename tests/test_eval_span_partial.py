"""Partial-credit policy: Jaccard goldens and greedy matching."""

import pytest

from slotvox.errors import ValidationError
from slotvox.eval.metrics import span_f1


def test_partial_credit_golden():
    gold = (("city", 0, 2), ("day", 3, 4))
    pred = (("city", 0, 2), ("day", 3, 5))
    score = span_f1(gold, pred, "partial")
    assert score.credit == pytest.approx(1.5)  # 1.0 exact + 1/2 Jaccard
    assert score.precision == pytest.approx(0.75)
    assert score.recall == pytest.approx(0.75)
    assert score.f1 == pytest.approx(0.75)


def test_labels_must_match_for_credit():
    score = span_f1((("city", 0, 2),), (("time", 0, 2),), "partial")
    assert score.credit == 0.0
    assert score.f1 == 0.0


def test_greedy_best_match_on_split_span():
    score = span_f1((("city", 0, 4),), (("city", 0, 2), ("city", 3, 4)), "partial")
    assert score.credit == pytest.approx(0.5)  # best pair: overlap 2 / union 4
    assert score.precision == pytest.approx(0.25)
    assert score.recall == pytest.approx(0.5)
    assert score.f1 == pytest.approx(1.0 / 3.0)


def test_strict_ignores_overlap():
    assert span_f1((("city", 0, 4),), (("city", 0, 2),)).credit == 0.0


def test_policy_validation():
    with pytest.raises(ValidationError, match="policy"):
        span_f1((), (), "loose")
