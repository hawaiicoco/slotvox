"""Vocab contracts and golden vocabularies for the weather domain.

Torch-free: vocabularies are pure label algebra.
"""

import pytest

from slotvox.errors import ValidationError
from slotvox.models.vocab import PAD_INDEX, Vocab, build_intent_vocab, build_tag_vocab
from slotvox.schema.builtins import weather_domain

DOMAIN = weather_domain()


def test_golden_intent_vocab():
    assert build_intent_vocab(DOMAIN) == ("query-weather", "query-forecast", "weather-alert")


def test_golden_tag_vocab():
    assert build_tag_vocab(DOMAIN) == (
        "O",
        "B-city",
        "I-city",
        "B-day",
        "I-day",
        "B-clock",
        "I-clock",
        "B-metric",
        "I-metric",
    )


def test_builders_require_domains():
    with pytest.raises(ValidationError, match="DomainSpec"):
        build_intent_vocab("weather")
    with pytest.raises(ValidationError, match="DomainSpec"):
        build_tag_vocab(None)


def test_vocab_round_trips_both_directions():
    vocab = Vocab(build_tag_vocab(DOMAIN))
    assert len(vocab) == 9
    for index, name in enumerate(vocab.names):
        assert vocab.name(index) == name
        assert vocab.index(name) == index
    assert vocab.indices(["O", "B-city"]) == (0, 1)


def test_vocab_unknown_lookups_rejected():
    vocab = Vocab(["a", "b"])
    with pytest.raises(ValidationError, match="unknown vocab entry"):
        vocab.index("c")
    with pytest.raises(ValidationError, match="out of range"):
        vocab.name(2)
    with pytest.raises(ValidationError, match="out of range"):
        vocab.name(-1)
    with pytest.raises(ValidationError):
        vocab.name(0.0)


def test_vocab_construction_rules():
    for bad in ([], "ab", 42, ["a", "a"], ["a", ""], [1]):
        with pytest.raises(ValidationError):
            Vocab(bad)


def test_vocab_serde_and_equality():
    vocab = Vocab(["a", "b"])
    assert Vocab.from_list(vocab.to_list()) == vocab
    assert vocab != Vocab(["b", "a"])
    assert vocab != "ab"
    with pytest.raises(ValidationError):
        Vocab.from_list(("a",))


def test_pad_index_is_the_ignore_label():
    assert PAD_INDEX == -100
