"""tags_to_spans and Span contracts."""

import pytest

from slotvox.errors import TaggingError
from slotvox.tagging.bio import Span, tags_to_spans


def test_single_span_with_inside_tags():
    assert tags_to_spans(("B-x", "I-x", "O")) == (Span("x", 0, 2),)


def test_adjacent_same_label_spans_stay_separate():
    assert tags_to_spans(("B-x", "B-x")) == (Span("x", 0, 1), Span("x", 1, 2))


def test_span_running_to_sequence_end():
    assert tags_to_spans(("O", "B-x", "I-x")) == (Span("x", 1, 3),)


def test_bioes_extraction():
    assert tags_to_spans(("S-x", "O", "B-y", "E-y")) == (Span("x", 0, 1), Span("y", 2, 4))


def test_empty_sequence_yields_no_spans():
    assert tags_to_spans(()) == ()
    assert tags_to_spans(("O", "O")) == ()


def test_invalid_sequences_raise():
    with pytest.raises(TaggingError):
        tags_to_spans(("I-x",))
    with pytest.raises(TaggingError):
        tags_to_spans("B-x")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"label": "x", "start": 2, "end": 2},
        {"label": "x", "start": 3, "end": 2},
        {"label": "x", "start": -1, "end": 2},
        {"label": "X", "start": 0, "end": 1},
        {"label": "x", "start": 0.0, "end": 1},
    ],
)
def test_invalid_spans_rejected(kwargs):
    with pytest.raises(TaggingError):
        Span(**kwargs)


def test_span_length_property():
    assert Span("x", 1, 4).n_tokens == 3
