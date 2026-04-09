"""TurnRecord contracts, joint turn accuracy, and micro pooling."""

import pytest

from slotvox.errors import ValidationError
from slotvox.eval.metrics import TurnRecord, micro_slot_f1, slice_pairs, turn_accuracy


def make(gi="a", pi="a", gt=("B-city", "O"), pt=("B-city", "O"), **kwargs):
    return TurnRecord(gi, pi, gt, pt, **kwargs)


def test_slice_pairs_canonical():
    assert slice_pairs({"noise": "clean", "domain": "weather"}) == (
        ("domain", "weather"),
        ("noise", "clean"),
    )
    with pytest.raises(ValidationError):
        slice_pairs({"Bad Kind": "x"})
    with pytest.raises(ValidationError):
        slice_pairs({"noise": ""})
    with pytest.raises(ValidationError, match="mapping"):
        slice_pairs("nope")


def test_record_validation_and_normalization():
    record = make(slices=slice_pairs({"noise": "clean"}))
    assert record.slices == (("noise", "clean"),)
    assert record.slice_mapping == {"noise": "clean"}
    assert TurnRecord("a", "a", ["O"], ["O"]).gold_tags == ("O",)
    with pytest.raises(ValidationError, match="whitespace"):
        TurnRecord("a b", "a", (), ())
    with pytest.raises(ValidationError, match="duplicate"):
        TurnRecord("a", "a", (), (), slices=(("noise", "x"), ("noise", "y")))
    with pytest.raises(ValidationError, match="pair"):
        TurnRecord("a", "a", (), (), slices=(("noise",),))


def test_turn_accuracy_is_strict_joint():
    records = [
        make(),
        make(pi="b"),
        make(pt=("O", "B-city")),
    ]
    score = turn_accuracy(records)
    assert (score.correct, score.total) == (1, 3)
    assert score.accuracy == pytest.approx(1 / 3)
    with pytest.raises(ValidationError, match="non-empty"):
        turn_accuracy(())
    with pytest.raises(ValidationError, match="TurnRecord"):
        turn_accuracy(["nope"])


def test_micro_slot_f1_pools_credit():
    first = make(gt=("B-city", "O", "B-day", "O"), pt=("B-city", "O", "B-day", "O"))
    second = make(gt=("B-city", "O"), pt=("O", "O"))
    score = micro_slot_f1([first, second])
    assert (score.gold_count, score.pred_count, score.credit) == (3, 2, 2.0)
    assert score.precision == 1.0
    assert score.recall == pytest.approx(2 / 3)
    partial = micro_slot_f1([first, second], "partial")
    assert partial.credit == score.credit
    with pytest.raises(ValidationError, match="policy"):
        micro_slot_f1([first], "loose")
