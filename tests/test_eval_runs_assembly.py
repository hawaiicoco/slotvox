"""ScoredRun.evaluate: every score from one record list."""

import pytest

from slotvox.errors import ValidationError
from slotvox.eval.metrics import TurnRecord
from slotvox.eval.runs import ScoredRun


def records():
    return (
        TurnRecord("a", "a", ("B-city", "O"), ("B-city", "O"), slices=(("noise", "clean"),)),
        TurnRecord("a", "b", ("B-city", "O"), ("O", "O"), slices=(("noise", "snr-15"),)),
        TurnRecord("b", "b", ("O",), ("O",), slices=(("noise", "snr-15"),)),
    )


def test_evaluate_computes_all_scores():
    run = ScoredRun.evaluate("run-one", records())
    assert run.record_count == 3
    assert (run.intent.correct, run.intent.total) == (2, 3)
    assert run.slot_strict.credit == 1.0
    assert (run.slot_strict.gold_count, run.slot_strict.pred_count) == (2, 1)
    assert run.slot_partial.credit == 1.0
    assert (run.turn.correct, run.turn.total) == (2, 3)
    assert run.confusion.labels == ("a", "b")
    assert run.confusion.counts == ((1, 1), (0, 1))
    assert run.bootstrap is None
    assert set(run.slices) == {"noise"}
    assert run.metadata == {}


def test_metadata_and_bootstrap_options():
    run = ScoredRun.evaluate(
        "run-two", records(), metadata={"model_hash": "f" * 64}, bootstrap_seed=11, replicates=50
    )
    assert run.metadata == {"model_hash": "f" * 64}
    assert run.bootstrap is not None
    assert run.bootstrap.replicates == 50
    with pytest.raises(ValidationError, match="JSON-native"):
        ScoredRun.evaluate("run-three", records(), metadata={"bad": object()})


def test_evaluate_validation():
    with pytest.raises(ValidationError, match="non-empty"):
        ScoredRun.evaluate("run-x", ())
    with pytest.raises(ValidationError):
        ScoredRun.evaluate("Bad Id", records())
    with pytest.raises(ValidationError, match="TurnRecord"):
        ScoredRun.evaluate("run-y", ["nope"])
