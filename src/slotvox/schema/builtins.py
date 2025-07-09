"""Built-in dialogue domains for synthetic experiments.

Hand-written task definitions (slot vocabularies + intents) used by the
data factory, CLI, and examples. They are schema content, not data: no
utterances live here, and nothing is tied to a real product or corpus.
"""

from __future__ import annotations

from slotvox.errors import ValidationError
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


def navigation_domain() -> DomainSpec:
    """Point-to-point navigation and nearby search."""
    slots = (
        SlotTypeSpec("destination", "text", {"min_length": 1, "max_length": 64}, "目的地"),
        SlotTypeSpec("origin", "text", {"min_length": 1, "max_length": 64}, "出发地"),
        SlotTypeSpec(
            "transport", "enum", {"choices": ["walk", "drive", "transit", "bike"]}, "出行方式"
        ),
        SlotTypeSpec(
            "category", "enum", {"choices": ["food", "fuel", "parking", "hotel"]}, "地点类别"
        ),
    )
    intents = (
        IntentDefinition(
            "navigate-to",
            (SlotRef("destination"), SlotRef("origin", True), SlotRef("transport", True)),
            "导航",
        ),
        IntentDefinition(
            "find-route",
            (SlotRef("origin"), SlotRef("destination"), SlotRef("transport", True)),
            "查路线",
        ),
        IntentDefinition("search-nearby", (SlotRef("category"),), "附近搜索"),
    )
    return DomainSpec(domain="navigation", slots=slots, intents=intents, description="导航出行域")


def calendar_domain() -> DomainSpec:
    """Calendar events, listing, cancellation, and reminders."""
    slots = (
        SlotTypeSpec("event-title", "text", {"min_length": 1, "max_length": 64}, "事件标题"),
        SlotTypeSpec("start-time", "time", description="开始时间"),
        SlotTypeSpec("end-time", "time", description="结束时间"),
        SlotTypeSpec("day", "enum", {"choices": ["今天", "明天", "后天", "周末"]}, "日期"),
        SlotTypeSpec("participant", "text", {"min_length": 1, "max_length": 32}, "参与人"),
    )
    intents = (
        IntentDefinition(
            "create-event",
            (
                SlotRef("event-title"),
                SlotRef("start-time"),
                SlotRef("end-time", True),
                SlotRef("participant", True),
            ),
            "创建日程",
        ),
        IntentDefinition("list-events", (SlotRef("day", True),), "查询日程"),
        IntentDefinition("cancel-event", (SlotRef("event-title"),), "取消日程"),
        IntentDefinition(
            "set-reminder",
            (SlotRef("event-title"), SlotRef("start-time"), SlotRef("day", True)),
            "设置提醒",
        ),
    )
    return DomainSpec(domain="calendar", slots=slots, intents=intents, description="日程管理域")


BUILTIN_DOMAIN_IDS = ("calendar", "music-control", "navigation", "weather")

_BUILDERS = {
    "calendar": calendar_domain,
    "music-control": music_control_domain,
    "navigation": navigation_domain,
    "weather": weather_domain,
}


def builtin_domain(domain_id: str) -> DomainSpec:
    """Return one of the built-in domains by id (strict)."""
    builder = _BUILDERS.get(domain_id) if isinstance(domain_id, str) else None
    if builder is None:
        raise ValidationError(
            f"unknown built-in domain {domain_id!r}; expected one of {BUILTIN_DOMAIN_IDS}"
        )
    return builder()


def builtin_domains() -> tuple[DomainSpec, ...]:
    """All built-in domains, ordered by id."""
    return tuple(builtin_domain(domain_id) for domain_id in BUILTIN_DOMAIN_IDS)
