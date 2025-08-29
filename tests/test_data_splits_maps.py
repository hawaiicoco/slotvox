"""speaker_splits/pattern_splits/check_policy/check_config contracts."""

import pytest

from slotvox.config.generation import GenerationConfig
from slotvox.data.splits import (
    SPLIT_POLICIES,
    check_config,
    check_policy,
    pattern_splits,
    speaker_splits,
)
from slotvox.errors import ValidationError

COUNTS = {"train": 6, "dev": 3, "test": 2}


def test_speaker_map_is_total_and_disjoint():
    mapping = speaker_splits(6, COUNTS, 9)
    assert set(mapping) == set(range(6))
    groups = {split: set() for split in COUNTS}
    for speaker, split in mapping.items():
        groups[split].add(speaker)
    assert groups["train"] & groups["dev"] == set()
    assert groups["train"] & groups["test"] == set()
    assert groups["dev"] & groups["test"] == set()


def test_speaker_map_depends_on_seed():
    assert speaker_splits(8, COUNTS, 1) != speaker_splits(8, COUNTS, 2)
    assert speaker_splits(8, COUNTS, 1) == speaker_splits(8, COUNTS, 1)


def test_insufficient_speakers_rejected():
    with pytest.raises(ValidationError, match="unique speaker"):
        speaker_splits(2, COUNTS, 5)
    for bad in (0, -1, True, 3.0):
        with pytest.raises(ValidationError):
            speaker_splits(bad, COUNTS, 5)


def test_pattern_map_is_total_and_disjoint():
    ids = ["a/1/0", "a/2/0", "a/3/0", "a/4/0"]
    mapping = pattern_splits(ids, COUNTS, 5)
    assert set(mapping) == set(ids)
    assert len(set(mapping.values())) == 3


def test_check_policy_both_directions():
    for policy in SPLIT_POLICIES:
        assert check_policy(policy) == policy
    with pytest.raises(ValidationError, match="unknown split policy"):
        check_policy("random")


def test_check_config_both_directions():
    config = GenerationConfig()
    assert check_config(config) is config
    with pytest.raises(ValidationError, match="GenerationConfig"):
        check_config({"kind": "generation"})
