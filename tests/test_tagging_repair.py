"""repair_sequence contracts: policies, identity, exhaustive validity."""

import itertools

import pytest

from slotvox.errors import TaggingError
from slotvox.tagging.bio import is_valid_sequence, repair_sequence

ALPHABET = ["O", "B-x", "I-x", "I-y", "B-y"]


def all_sequences(max_length=3):
    for length in range(max_length + 1):
        yield from itertools.product(ALPHABET, repeat=length)


def test_strict_policy_passes_valid_and_raises_invalid():
    assert repair_sequence(("B-x", "I-x"), "strict") == ("B-x", "I-x")
    with pytest.raises(TaggingError):
        repair_sequence(("I-x",), "strict")


def test_drop_replaces_violations_with_outside():
    assert repair_sequence(("I-y",), "drop") == ("O",)
    assert repair_sequence(("B-x", "I-y", "I-x"), "drop") == ("B-x", "O", "O")


def test_promote_restarts_spans():
    assert repair_sequence(("I-y",), "promote") == ("B-y",)
    assert repair_sequence(("B-x", "I-y", "I-y"), "promote") == ("B-x", "B-y", "I-y")


def test_identity_on_valid_sequences_exhaustively():
    for seq in all_sequences(3):
        if not is_valid_sequence(seq):
            continue
        assert repair_sequence(seq, "drop") == seq
        assert repair_sequence(seq, "promote") == seq


def test_repaired_always_valid_exhaustively():
    repaired_count = 0
    for seq in all_sequences(3):
        for policy in ("drop", "promote"):
            repaired = repair_sequence(seq, policy)
            assert len(repaired) == len(seq)
            assert is_valid_sequence(repaired)
            if repaired != seq:
                repaired_count += 1
    assert repaired_count > 0


def test_unknown_policy_and_bioes_rejected():
    with pytest.raises(TaggingError, match="unknown repair policy"):
        repair_sequence(("O",), "fix")
    with pytest.raises(TaggingError, match="BIO tags"):
        repair_sequence(("S-x",), "drop")
