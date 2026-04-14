"""Paired bootstrap CIs: determinism, degenerate cases, validation."""

import pytest

from slotvox.errors import ValidationError
from slotvox.eval.bootstrap import (
    BootstrapCI,
    intent_statistic,
    paired_bootstrap_ci,
    slot_strict_statistic,
    turn_statistic,
)
from slotvox.eval.metrics import TurnRecord


def records(n_correct=6, n_wrong=2):
    items = [TurnRecord("a", "a", ("B-city", "O"), ("B-city", "O")) for _ in range(n_correct)]
    items += [TurnRecord("a", "b", ("B-city", "O"), ("O", "O")) for _ in range(n_wrong)]
    return tuple(items)


def test_same_seed_identical_different_seed_differs():
    # percentile endpoints are coarse on tiny samples, so seed sensitivity
    # is asserted on a larger fixture where endpoints actually move
    items = records(20, 7)
    first = paired_bootstrap_ci(items, seed=101, replicates=200)
    second = paired_bootstrap_ci(items, seed=101, replicates=200)
    assert first == second
    third = paired_bootstrap_ci(items, seed=202, replicates=200)
    assert (third.low, third.high) != (first.low, first.high)


def test_estimate_and_interval_sane():
    ci = paired_bootstrap_ci(records(), seed=7, replicates=200)
    assert ci.estimate == 0.75
    assert ci.low <= ci.estimate <= ci.high
    assert 0.0 <= ci.low
    assert ci.high <= 1.0
    assert ci.replicates == 200
    assert ci.confidence == 0.95


def test_degenerate_all_correct_has_zero_width():
    ci = paired_bootstrap_ci(records(4, 0), intent_statistic, seed=3, replicates=50)
    assert ci.estimate == ci.low == ci.high == 1.0


def test_statistics_and_serde():
    items = records()
    assert intent_statistic(items) == 0.75
    assert turn_statistic(items) == 0.75  # the two wrong-intent records fail the joint check
    assert 0.0 <= slot_strict_statistic(items) <= 1.0
    ci = paired_bootstrap_ci(items, seed=5, replicates=50)
    assert BootstrapCI.from_dict(ci.to_dict()) == ci
    with pytest.raises(ValidationError, match="unknown keys"):
        BootstrapCI.from_dict({**ci.to_dict(), "extra": 1})


def test_validation():
    with pytest.raises(ValidationError, match="replicates"):
        paired_bootstrap_ci(records(), seed=1, replicates=5)
    with pytest.raises(ValidationError, match="confidence"):
        paired_bootstrap_ci(records(), seed=1, replicates=20, confidence=1.5)
    with pytest.raises(ValidationError, match="non-empty"):
        paired_bootstrap_ci((), seed=1)
    with pytest.raises(ValidationError, match="callable"):
        paired_bootstrap_ci(records(), "not-callable", seed=1)
    with pytest.raises(ValidationError, match="low"):
        BootstrapCI(0.5, 0.9, 0.1, 100, 0.95, 1)
    with pytest.raises(ValidationError, match="seed"):
        BootstrapCI(0.5, 0.1, 0.9, 100, 0.95, -1)
