"""Contracts for make_rng and derive_seed."""

import numpy as np
import pytest

from slotvox.errors import ValidationError
from slotvox.util.seed import SEED_MAX, derive_seed, make_rng


def test_same_seed_produces_same_stream():
    assert np.array_equal(make_rng(7).standard_normal(8), make_rng(7).standard_normal(8))


def test_different_seeds_produce_different_streams():
    assert not np.array_equal(make_rng(7).standard_normal(8), make_rng(8).standard_normal(8))


@pytest.mark.parametrize("bad", [-1, SEED_MAX + 1, 1.5, "7", True, None])
def test_invalid_seeds_rejected(bad):
    with pytest.raises(ValidationError):
        make_rng(bad)


def test_boundary_seeds_accepted():
    assert make_rng(0) is not None
    assert make_rng(SEED_MAX) is not None


def test_derive_seed_is_deterministic_and_bounded():
    first = derive_seed("split", 3, {"x": [1, 2]})
    assert first == derive_seed("split", 3, {"x": [1, 2]})
    assert 0 <= first <= SEED_MAX


def test_derive_seed_separates_parts():
    assert derive_seed("train", 1) != derive_seed("dev", 1)
    assert derive_seed("x", {"a": 1}) != derive_seed("x", {"a": 2})


def test_derive_seed_requires_parts():
    with pytest.raises(ValidationError):
        derive_seed()
