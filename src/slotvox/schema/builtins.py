"""Built-in dialogue domains for synthetic experiments.

Hand-written task definitions (slot vocabularies + intents) used by the
data factory, CLI, and examples. They are schema content, not data: no
utterances live here, and nothing is tied to a real product or corpus.
"""

from __future__ import annotations

from slotvox.schema.domains import DomainSpec
from slotvox.schema.intents import IntentDefinition, SlotRef
from slotvox.schema.slots import SlotTypeSpec


def weather_domain() -> DomainSpec:
    """Weather queries over a small city enum with day/metric refinements."""
    slots = (
        SlotTypeSpec("city", "enum", {"choices": ["北京", "上海", "广州", "深圳", "杭州"]}, "城市"),
        SlotTypeSpec("day", "enum", {"choices": ["今天", "明天", "后天", "周末"]}, "日期"),
        SlotTypeSpec("clock", "time", description="时间点 HH:MM"),
        SlotTypeSpec(
            "metric",
            "enum",
            {"choices": ["temperature", "rain", "wind", "humidity"]},
            "指标",
        ),
    )
    intents = (
        IntentDefinition(
            "query-weather",
            (SlotRef("city"), SlotRef("day", True), SlotRef("metric", True)),
            "查询天气",
        ),
        IntentDefinition(
            "query-forecast",
            (SlotRef("city", True), SlotRef("day"), SlotRef("clock", True)),
            "查询预报",
        ),
        IntentDefinition("weather-alert", (SlotRef("city", True),), "天气预警"),
    )
    return DomainSpec(domain="weather", slots=slots, intents=intents, description="天气查询域")


def music_control_domain() -> DomainSpec:
    """Music playback control with song/artist/genre/playlist slots."""
    slots = (
        SlotTypeSpec("song", "text", {"min_length": 1, "max_length": 64}, "歌名"),
        SlotTypeSpec("artist", "text", {"min_length": 1, "max_length": 64}, "歌手"),
        SlotTypeSpec(
            "genre",
            "enum",
            {"choices": ["pop", "rock", "jazz", "classical", "electronic"]},
            "流派",
        ),
        SlotTypeSpec("volume-level", "number", {"minimum": 0, "maximum": 100}, "音量"),
        SlotTypeSpec("playlist", "text", {"min_length": 1, "max_length": 64}, "歌单"),
    )
    intents = (
        IntentDefinition(
            "play-music",
            (
                SlotRef("song", True),
                SlotRef("artist", True),
                SlotRef("genre", True),
                SlotRef("playlist", True),
            ),
            "播放音乐",
        ),
        IntentDefinition("pause-music", (), "暂停"),
        IntentDefinition("resume-music", (), "继续播放"),
        IntentDefinition("stop-music", (), "停止"),
        IntentDefinition("next-track", (), "下一首"),
        IntentDefinition("previous-track", (), "上一首"),
        IntentDefinition("set-volume", (SlotRef("volume-level"),), "设置音量"),
    )
    return DomainSpec(
        domain="music-control", slots=slots, intents=intents, description="音乐控制域"
    )
