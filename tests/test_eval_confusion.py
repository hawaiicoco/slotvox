"""Confusion matrix contracts and per-label goldens."""

import pytest

from slotvox.errors import ValidationError
from slotvox.eval.confusion import ConfusionMatrix


def test_from_pairs_golden():
    matrix = ConfusionMatrix.from_pairs(("a", "a", "b"), ("a", "b", "b"))
    assert matrix.labels == ("a", "b")
    assert matrix.counts == ((1, 1), (0, 1))
    assert matrix.total == 3
    assert matrix.accuracy == pytest.approx(2 / 3)


def test_explicit_labels_and_zero_rows():
    matrix = ConfusionMatrix.from_pairs(("a",), ("a",), labels=("a", "c"))
    assert matrix.labels == ("a", "c")
    assert matrix.counts == ((1, 0), (0, 0))
    stats = matrix.per_label()
    assert stats["c"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    assert stats["a"]["support"] == 1


def test_per_label_golden():
    matrix = ConfusionMatrix.from_pairs(("a", "a", "b"), ("a", "b", "b"))
    stats = matrix.per_label()
    assert list(stats) == ["a", "b"]
    # a: tp 1, predicted 1, gold 2 -> p 1.0 r 0.5; b: tp 1, predicted 2, gold 1
    assert stats["a"] == {
        "precision": 1.0,
        "recall": 0.5,
        "f1": pytest.approx(2 / 3),
        "support": 2,
    }
    assert stats["b"] == {
        "precision": 0.5,
        "recall": 1.0,
        "f1": pytest.approx(2 / 3),
        "support": 1,
    }


def test_rejections():
    with pytest.raises(ValidationError, match="align"):
        ConfusionMatrix.from_pairs(("a",), ())
    with pytest.raises(ValidationError, match="unknown labels"):
        ConfusionMatrix.from_pairs(("a",), ("b",), labels=("a",))
    with pytest.raises(ValidationError, match="unknown labels"):
        ConfusionMatrix.from_pairs(("a", "z"), ("a", "a"), labels=("a",))
    with pytest.raises(ValidationError, match="at least"):
        ConfusionMatrix.from_pairs((), ())
    empty = ConfusionMatrix.from_pairs((), (), labels=("a",))
    assert empty.accuracy == 0.0


def test_serde_roundtrip_and_shape_validation():
    matrix = ConfusionMatrix.from_pairs(("a", "b"), ("a", "a"))
    payload = matrix.to_dict()
    assert sorted(payload) == ["counts", "labels"]
    assert ConfusionMatrix.from_dict(payload) == matrix
    with pytest.raises(ValidationError, match="exactly"):
        ConfusionMatrix.from_dict({**payload, "extra": 1})
    with pytest.raises(ValidationError, match="square"):
        ConfusionMatrix(("a", "b"), ((1, 0),))
    with pytest.raises(ValidationError, match="columns"):
        ConfusionMatrix(("a", "b"), ((1, 0, 0), (0, 1, 0)))
    with pytest.raises(ValidationError, match="non-negative"):
        ConfusionMatrix(("a",), ((-1,),))
    with pytest.raises(ValidationError, match="unique"):
        ConfusionMatrix(("a", "a"), ((1, 0), (0, 1)))
