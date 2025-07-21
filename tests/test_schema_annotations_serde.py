"""AnnotatedUtterance serialization contracts, including a golden dict."""

import pytest

from slotvox.errors import ValidationError
from slotvox.schema.annotations import AnnotatedUtterance
from slotvox.util.jsoncanon import stable_hash


def make():
    return AnnotatedUtterance(
        utterance_id="utt-0",
        language="zh",
        tokens=("北京", "明天", "天气"),
        tags=("B-city", "B-day", "O"),
        intent="query-weather",
    )


def test_round_trip_preserves_equality():
    assert AnnotatedUtterance.from_dict(make().to_dict()) == make()


def test_golden_annotation_dict():
    assert make().to_dict() == {
        "utterance_id": "utt-0",
        "language": "zh",
        "tokens": ["北京", "明天", "天气"],
        "tags": ["B-city", "B-day", "O"],
        "intent": "query-weather",
    }
    assert isinstance(stable_hash(make().to_dict()), str)


def test_from_dict_strict_keys():
    payload = make().to_dict()
    with pytest.raises(ValidationError, match="unknown keys"):
        AnnotatedUtterance.from_dict({**payload, "extra": 1})
    with pytest.raises(ValidationError, match="missing keys"):
        AnnotatedUtterance.from_dict({k: v for k, v in payload.items() if k != "tags"})
    with pytest.raises(ValidationError, match="must be lists"):
        AnnotatedUtterance.from_dict({**payload, "tokens": "北京"})


def test_from_dict_reruns_full_validation():
    payload = make().to_dict()
    payload["tags"] = ["X-city", "O", "O"]
    with pytest.raises(ValidationError):
        AnnotatedUtterance.from_dict(payload)
