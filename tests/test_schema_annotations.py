"""AnnotatedUtterance structural validation contracts."""

import pytest

from slotvox.errors import ValidationError
from slotvox.schema.annotations import AnnotatedUtterance


def make(**overrides):
    data = {
        "utterance_id": "utt-0",
        "language": "zh",
        "tokens": ("北京", "明天", "天气"),
        "tags": ("B-city", "B-day", "O"),
        "intent": "query-weather",
    }
    data.update(overrides)
    return AnnotatedUtterance(**data)


def test_text_joins_per_language():
    assert make().text == "北京明天天气"
    english = make(language="en", tokens=("what", "time"), tags=("O", "O"), intent="list-events")
    assert english.text == "what time"


def test_tag_slot_names_sorted_unique():
    ann = make(tokens=("北京", "市", "明天", "天气"), tags=("B-city", "I-city", "B-day", "O"))
    assert ann.tag_slot_names() == ("city", "day")


def test_tag_misalignment_rejected():
    with pytest.raises(ValidationError, match="align"):
        make(tags=("O", "O"))


@pytest.mark.parametrize(
    "tags",
    [
        ("X-city", "O", "O"),
        ("B-", "O", "O"),
        ("B-Bad Slot", "O", "O"),
        ("city", "O", "O"),
        ("", "O", "O"),
    ],
)
def test_invalid_tags_rejected(tags):
    with pytest.raises(ValidationError):
        make(tags=tags)


def test_invalid_tokens_rejected():
    with pytest.raises(ValidationError):
        make(tokens=())
    with pytest.raises(ValidationError):
        make(tokens=("", "x"))
    with pytest.raises(ValidationError):
        make(tokens=("has space", "x"))
    with pytest.raises(ValidationError):
        make(tokens=tuple("x" for _ in range(1001)), tags=tuple("O" for _ in range(1001)))


def test_invalid_identity_fields_rejected():
    with pytest.raises(ValidationError):
        make(utterance_id="")
    with pytest.raises(ValidationError):
        make(utterance_id="has space")
    with pytest.raises(ValidationError):
        make(utterance_id="x" * 129)
    with pytest.raises(ValidationError):
        make(language="fr")
    with pytest.raises(ValidationError):
        make(intent="Query")
