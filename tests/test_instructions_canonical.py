"""Canonical system/assistant templates and slot rendering goldens."""

import pytest

from slotvox.errors import ValidationError
from slotvox.instructions.templates import (
    ASSISTANT_TEMPLATES,
    EMPTY_SLOTS,
    SYSTEM_TEMPLATES,
    slots_text,
)

WEATHER_SLOTS = ("city", "day", "clock", "metric")


def test_system_templates_golden():
    assert SYSTEM_TEMPLATES["zh"].render({"domain": "weather"}).text == (
        "你是语音助手。听用户的语音，识别weather领域的意图和槽位。"
    )
    assert SYSTEM_TEMPLATES["en"].render({"domain": "weather"}).text == (
        "You are a speech assistant. Listen to the user's audio and identify "
        "the intent and slots for the weather domain."
    )


def test_assistant_templates_golden():
    variables = {"intent": "query-weather", "slots": "city=上海, day=今天"}
    assert ASSISTANT_TEMPLATES["zh"].render(variables).text == (
        "意图：query-weather；槽位：city=上海, day=今天"
    )
    assert ASSISTANT_TEMPLATES["en"].render(variables).text == (
        "Intent: query-weather; Slots: city=上海, day=今天"
    )
    assert EMPTY_SLOTS == {"zh": "无", "en": "none"}


def test_slots_text_domain_order_and_joining():
    assert slots_text({"day": ("今天",), "city": ("上海",)}, WEATHER_SLOTS) == (
        "city=上海, day=今天"
    )
    assert slots_text({"city": ("北京", "上海")}, WEATHER_SLOTS) == "city=北京/上海"
    assert slots_text({}, WEATHER_SLOTS) == ""


def test_slots_text_rejections():
    with pytest.raises(ValidationError, match="unknown slots"):
        slots_text({"artist": ("x",)}, WEATHER_SLOTS)
    with pytest.raises(ValidationError, match="non-empty"):
        slots_text({"city": ()}, WEATHER_SLOTS)
    with pytest.raises(ValidationError, match="non-blank"):
        slots_text({"city": ("  ",)}, WEATHER_SLOTS)
    with pytest.raises(ValidationError, match="braces"):
        slots_text({"city": ("{x}",)}, WEATHER_SLOTS)
    with pytest.raises(ValidationError, match="mapping"):
        slots_text([("city", ("x",))], WEATHER_SLOTS)
