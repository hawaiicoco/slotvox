"""Exhaustive small-case roundtrip: tags -> spans -> tags is identity."""

import itertools

import pytest

from slotvox.errors import TaggingError
from slotvox.tagging.bio import Span, is_valid_sequence, spans_to_tags, tags_to_spans

ALPHABET = ["O", "B-x", "I-x", "B-y", "I-y"]


def all_sequences(alphabet, max_length=3):
    for length in range(max_length + 1):
        yield from itertools.product(alphabet, repeat=length)


def test_valid_sequences_round_trip_exhaustively():
    checked = 0
    for seq in all_sequences(ALPHABET, 3):
        if not is_valid_sequence(seq):
            continue
        assert spans_to_tags(tags_to_spans(seq), len(seq)) == seq
        checked += 1
    assert checked >= 50


def test_spans_to_tags_rejects_bad_inputs():
    with pytest.raises(TaggingError, match="overlap"):
        spans_to_tags([Span("x", 0, 2), Span("y", 1, 3)], 4)
    with pytest.raises(TaggingError, match="sorted"):
        spans_to_tags([Span("y", 2, 3), Span("x", 0, 1)], 4)
    with pytest.raises(TaggingError, match="exceeds"):
        spans_to_tags([Span("x", 0, 5)], 4)
    with pytest.raises(TaggingError, match="n_tokens"):
        spans_to_tags([], -1)
    with pytest.raises(TaggingError, match="Span instances"):
        spans_to_tags(["x"], 1)


def test_spans_to_tags_bioes_mode():
    tags = spans_to_tags([Span("x", 0, 1), Span("y", 1, 4)], 4, bioes=True)
    assert tags == ("S-x", "B-y", "I-y", "E-y")


def test_empty_spans_all_outside():
    assert spans_to_tags([], 3) == ("O", "O", "O")
    assert spans_to_tags((), 0) == ()
