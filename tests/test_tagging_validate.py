"""validate_sequence / is_valid_sequence contracts."""

import pytest

from slotvox.errors import TaggingError
from slotvox.tagging.bio import is_valid_sequence, validate_sequence

VALID_BIO = [
    (),
    ("O",),
    ("B-x", "I-x", "O"),
    ("B-x", "B-x"),
    ("B-x", "I-x", "B-y", "I-y"),
    ("O", "B-x", "I-x", "I-x"),
]

INVALID_BIO = [
    ("I-x",),
    ("O", "I-x"),
    ("B-x", "I-y"),
    ("B-x", "I-x", "I-y"),
]

VALID_BIOES = [
    (),
    ("O",),
    ("S-x",),
    ("B-x", "E-x"),
    ("B-x", "I-x", "E-x"),
    ("O", "S-x", "O", "B-y", "I-y", "E-y"),
]

INVALID_BIOES = [
    ("B-x", "I-x"),
    ("E-x",),
    ("I-x", "E-x"),
    ("B-x", "S-y"),
    ("B-x", "B-y", "E-y", "E-x"),
    ("S-x", "E-x"),
]


@pytest.mark.parametrize("seq", VALID_BIO)
def test_valid_bio_accepted(seq):
    validate_sequence(seq)
    assert is_valid_sequence(seq)


@pytest.mark.parametrize("seq", INVALID_BIO)
def test_invalid_bio_rejected(seq):
    with pytest.raises(TaggingError):
        validate_sequence(seq)
    assert not is_valid_sequence(seq)


@pytest.mark.parametrize("seq", VALID_BIOES)
def test_valid_bioes_accepted(seq):
    validate_sequence(seq, bioes=True)


@pytest.mark.parametrize("seq", INVALID_BIOES)
def test_invalid_bioes_rejected(seq):
    assert not is_valid_sequence(seq, bioes=True)


def test_bioes_tags_rejected_in_bio_mode():
    with pytest.raises(TaggingError, match="BIOES-only"):
        validate_sequence(("S-x",))
    with pytest.raises(TaggingError, match="BIOES-only"):
        validate_sequence(("B-x", "E-x"))


def test_non_sequence_inputs_rejected():
    with pytest.raises(TaggingError):
        validate_sequence("B-x")
    with pytest.raises(TaggingError):
        validate_sequence(42)
