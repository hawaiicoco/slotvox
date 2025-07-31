"""BIO<->BIOES conversion contracts, exhaustive on a small alphabet."""

import itertools

import pytest

from slotvox.errors import TaggingError
from slotvox.tagging.bio import bio_to_bioes, bioes_to_bio, is_valid_sequence

ALPHABET = ["O", "B-x", "I-x", "E-x", "S-x"]


def all_sequences(max_length=3):
    for length in range(max_length + 1):
        yield from itertools.product(ALPHABET, repeat=length)


def test_bio_to_bioes_shapes():
    assert bio_to_bioes(("B-x", "I-x", "O")) == ("B-x", "E-x", "O")
    assert bio_to_bioes(("B-x", "O")) == ("S-x", "O")
    assert bio_to_bioes(("B-x", "I-x", "I-x")) == ("B-x", "I-x", "E-x")


def test_bioes_to_bio_shapes():
    assert bioes_to_bio(("S-x", "O")) == ("B-x", "O")
    assert bioes_to_bio(("B-x", "E-x")) == ("B-x", "I-x")
    assert bioes_to_bio(("B-x", "I-x", "E-x")) == ("B-x", "I-x", "I-x")


def test_round_trips_exhaustively_both_directions():
    bio_count = bioes_count = 0
    for seq in all_sequences(3):
        if is_valid_sequence(seq, bioes=True):
            assert bio_to_bioes(bioes_to_bio(seq)) == seq
            bioes_count += 1
        if is_valid_sequence(seq):
            assert bioes_to_bio(bio_to_bioes(seq)) == seq
            bio_count += 1
    assert bioes_count >= 10
    assert bio_count >= 10


def test_invalid_inputs_raise():
    with pytest.raises(TaggingError):
        bio_to_bioes(("I-x",))
    with pytest.raises(TaggingError):
        bioes_to_bio(("B-x", "I-x"))
    with pytest.raises(TaggingError):
        bioes_to_bio(("E-x",))
    with pytest.raises(TaggingError):
        bio_to_bioes("B-x")
