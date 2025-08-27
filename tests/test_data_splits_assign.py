"""assign_keys_to_splits contracts: disjointness, determinism, rejection."""

import pytest

from slotvox.data.splits import assign_keys_to_splits
from slotvox.errors import ValidationError

COUNTS = {"train": 6, "dev": 3, "test": 2}


def test_keys_are_disjoint_and_covered():
    result = assign_keys_to_splits(list(range(5)), COUNTS, 7, name="speaker")
    assert set(result) == {"train", "dev", "test"}
    flat = [key for bucket in result.values() for key in bucket]
    assert len(flat) == len(set(flat)) == 5
    assert all(len(bucket) >= 1 for bucket in result.values())


def test_assignment_is_deterministic_per_seed():
    a = assign_keys_to_splits(list(range(8)), COUNTS, 11)
    b = assign_keys_to_splits(list(range(8)), COUNTS, 11)
    c = assign_keys_to_splits(list(range(8)), COUNTS, 12)
    assert a == b
    assert a != c


def test_empty_splits_receive_no_bucket():
    result = assign_keys_to_splits(["x", "y"], {"train": 4}, 3)
    assert set(result) == {"train"}
    assert sorted(result["train"]) == ["x", "y"]


def test_duplicate_keys_collapse():
    result = assign_keys_to_splits([1, 1, 2, 3], COUNTS, 5)
    flat = [key for bucket in result.values() for key in bucket]
    assert sorted(flat) == [1, 2, 3]


def test_insufficient_keys_rejected():
    with pytest.raises(ValidationError, match="unique"):
        assign_keys_to_splits([1, 2], COUNTS, 5)
    with pytest.raises(ValidationError, match="non-empty split"):
        assign_keys_to_splits([1], {"train": 0, "dev": 0}, 5)


def test_invalid_inputs_rejected():
    with pytest.raises(ValidationError):
        assign_keys_to_splits("abc", COUNTS, 5)
    with pytest.raises(ValidationError):
        assign_keys_to_splits([1], {"nope": 1}, 5)
    with pytest.raises(ValidationError):
        assign_keys_to_splits([1, 2], {"train": -1, "dev": 1}, 5)
    with pytest.raises(ValidationError):
        assign_keys_to_splits([1, 2, 3], COUNTS, -5)
