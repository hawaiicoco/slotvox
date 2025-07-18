"""AnnotatedUtterance.validate_against contracts, both directions."""

import pytest

from slotvox.errors import ValidationError
from slotvox.schema.annotations import AnnotatedUtterance
from slotvox.schema.builtins import weather_domain


def make(tokens, tags, intent="query-weather"):
    return AnnotatedUtterance(
        utterance_id="utt-0", language="zh", tokens=tuple(tokens), tags=tuple(tags), intent=intent
    )


DOMAIN = weather_domain()


def test_valid_annotation_passes():
    make(("北京", "明天", "天气"), ("B-city", "B-day", "O")).validate_against(DOMAIN)


def test_all_optional_slots_may_be_absent():
    make(("有", "预警", "吗"), ("O", "O", "O"), intent="weather-alert").validate_against(DOMAIN)


def test_unknown_intent_rejected():
    with pytest.raises(ValidationError, match="unknown intent"):
        make(("播放", "音乐"), ("O", "O"), intent="play-music").validate_against(DOMAIN)


def test_unknown_slot_label_rejected():
    with pytest.raises(ValidationError, match="unknown in domain"):
        make(("放", "一首", "歌"), ("O", "O", "B-song")).validate_against(DOMAIN)


def test_slot_outside_intent_rejected():
    with pytest.raises(ValidationError, match="does not take slot"):
        make(("明天", "天气"), ("B-day", "O"), intent="weather-alert").validate_against(DOMAIN)


def test_missing_required_slot_rejected():
    with pytest.raises(ValidationError, match="missing required slots"):
        make(("明天", "天气"), ("B-day", "O")).validate_against(DOMAIN)


def test_non_domain_argument_rejected():
    with pytest.raises(ValidationError, match="DomainSpec"):
        make(("北京",), ("B-city",)).validate_against("weather")
