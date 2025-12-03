"""Largest-remainder mixing quotas and their validation."""

import pytest

from slotvox.errors import ValidationError
from slotvox.instructions.mixing import mixing_quotas


def test_largest_remainder_goldens():
    assert mixing_quotas({"a": 1, "b": 1, "c": 1}, 5) == {"a": 2, "b": 2, "c": 1}
    assert mixing_quotas({"a": 3, "b": 1}, 8) == {"a": 6, "b": 2}
    assert mixing_quotas({"a": 1.0, "b": 2.0, "c": 3.0}, 7) == {"a": 1, "b": 2, "c": 4}


def test_zero_weight_gets_zero_quota():
    assert mixing_quotas({"a": 1, "b": 0}, 3) == {"a": 3, "b": 0}


def test_quotas_always_sum_to_size():
    for size in (1, 2, 13):
        assert sum(mixing_quotas({"a": 1.0, "b": 2.0, "c": 3.0}, size).values()) == size


def test_validation():
    with pytest.raises(ValidationError, match="non-empty"):
        mixing_quotas({}, 5)
    with pytest.raises(ValidationError, match="positive"):
        mixing_quotas({"a": 0}, 5)
    with pytest.raises(ValidationError, match="size"):
        mixing_quotas({"a": 1}, 0)
    with pytest.raises(ValidationError, match="weight"):
        mixing_quotas({"a": -1}, 5)
    with pytest.raises(ValidationError, match="weight"):
        mixing_quotas({"a": float("nan")}, 5)
    with pytest.raises(ValidationError):
        mixing_quotas({"Bad Src": 1}, 5)
