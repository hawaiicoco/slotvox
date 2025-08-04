"""Synthetic value lexicons for built-in domains.

A lexicon lists, per slot, the ``(surface, value)`` pairs the factory can
draw from: ``surface`` is the text as it appears in a transcription
(tokenized per language), ``value`` is the semantic slot value checked
against the domain's :class:`SlotTypeSpec`. English surfaces may map to the
same canonical values as Chinese ones (a surface form is not a value).

Everything here is invented fixture content for synthetic experiments —
not real user data, places, songs, or schedules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from slotvox.errors import ValidationError
from slotvox.schema.annotations import LANGUAGES
from slotvox.schema.builtins import builtin_domain
from slotvox.schema.domains import DomainSpec


@dataclass(frozen=True)
class LexiconEntry:
    """One (surface text, semantic value) pair for a slot."""

    surface: str
    value: Any

    def __post_init__(self) -> None:
        if not isinstance(self.surface, str) or not self.surface:
            raise ValidationError("LexiconEntry.surface must be a non-empty string")
        if self.surface != self.surface.strip():
            raise ValidationError(
                f"LexiconEntry.surface must not have edge whitespace: {self.surface!r}"
            )


Lexicon = dict[str, tuple[LexiconEntry, ...]]

_LEXICONS: dict[tuple[str, str], Lexicon] = {}


def validate_lexicon(domain: DomainSpec, lexicon: Any) -> None:
    """Validate a lexicon against a domain, strictly and in both directions.

    Every listed slot must exist in the domain, entries must be non-empty,
    surfaces unique per slot, and every value must satisfy the slot's
    constraint (``check_value``).
    """
    if not isinstance(domain, DomainSpec):
        raise ValidationError(f"validate_lexicon expects a DomainSpec, got {type(domain).__name__}")
    if not isinstance(lexicon, dict) or not lexicon:
        raise ValidationError("lexicon must be a non-empty dict of slot -> entries")
    for slot, entries in lexicon.items():
        spec = domain.slot_spec(slot)
        if not isinstance(entries, tuple) or not entries:
            raise ValidationError(f"lexicon slot {slot!r} must map to a non-empty tuple")
        surfaces: set[str] = set()
        for entry in entries:
            if not isinstance(entry, LexiconEntry):
                raise ValidationError(f"lexicon slot {slot!r} must contain LexiconEntry instances")
            spec.check_value(entry.value)
            if entry.surface in surfaces:
                raise ValidationError(
                    f"lexicon slot {slot!r} has duplicate surface {entry.surface!r}"
                )
            surfaces.add(entry.surface)


def register_lexicon(domain_id: str, language: str, lexicon: Lexicon) -> None:
    """Register (and validate) the lexicon for one domain and language."""
    domain = builtin_domain(domain_id)
    if language not in LANGUAGES:
        raise ValidationError(f"language must be one of {LANGUAGES}, got {language!r}")
    key = (domain_id, language)
    if key in _LEXICONS:
        raise ValidationError(f"lexicon already registered for {key}")
    validate_lexicon(domain, lexicon)
    _LEXICONS[key] = lexicon


def slot_entries(domain_id: str, language: str, slot: str) -> tuple[LexiconEntry, ...]:
    """Strict lookup of the entries for one slot."""
    key = (domain_id, language)
    lexicon = _LEXICONS.get(key)
    if lexicon is None:
        raise ValidationError(
            f"no lexicon registered for domain {domain_id!r} language {language!r}"
        )
    entries = lexicon.get(slot)
    if entries is None:
        raise ValidationError(
            f"lexicon for domain {domain_id!r} language {language!r} has no slot {slot!r}"
        )
    return entries


def registered_lexicons() -> tuple[tuple[str, str], ...]:
    """All (domain, language) keys with a registered lexicon, sorted."""
    return tuple(sorted(_LEXICONS))


register_lexicon(
    "weather",
    "zh",
    {
        "city": (
            LexiconEntry("北京", "北京"),
            LexiconEntry("上海", "上海"),
            LexiconEntry("广州", "广州"),
        ),
        "day": (
            LexiconEntry("今天", "今天"),
            LexiconEntry("明天", "明天"),
            LexiconEntry("周末", "周末"),
        ),
        "clock": (
            LexiconEntry("七点半", "07:30"),
            LexiconEntry("九点", "09:00"),
        ),
        "metric": (
            LexiconEntry("气温", "temperature"),
            LexiconEntry("下雨", "rain"),
            LexiconEntry("风力", "wind"),
        ),
    },
)


register_lexicon(
    "music-control",
    "zh",
    {
        "song": (
            LexiconEntry("青花瓷", "青花瓷"),
            LexiconEntry("稻香", "稻香"),
            LexiconEntry("夜曲", "夜曲"),
        ),
        "artist": (
            LexiconEntry("周杰伦", "周杰伦"),
            LexiconEntry("邓紫棋", "邓紫棋"),
        ),
        "genre": (
            LexiconEntry("流行", "pop"),
            LexiconEntry("摇滚", "rock"),
            LexiconEntry("爵士", "jazz"),
        ),
        "volume-level": (
            LexiconEntry("三十", 30),
            LexiconEntry("五十", 50),
            LexiconEntry("八十", 80),
        ),
        "playlist": (
            LexiconEntry("我的收藏", "我的收藏"),
            LexiconEntry("跑步歌单", "跑步歌单"),
        ),
    },
)


register_lexicon(
    "navigation",
    "zh",
    {
        "destination": (
            LexiconEntry("人民广场", "人民广场"),
            LexiconEntry("虹桥机场", "虹桥机场"),
            LexiconEntry("公司", "公司"),
        ),
        "origin": (
            LexiconEntry("家", "家"),
            LexiconEntry("学校", "学校"),
        ),
        "transport": (
            LexiconEntry("步行", "walk"),
            LexiconEntry("开车", "drive"),
            LexiconEntry("地铁", "transit"),
        ),
        "category": (
            LexiconEntry("餐厅", "food"),
            LexiconEntry("加油站", "fuel"),
            LexiconEntry("停车场", "parking"),
        ),
    },
)


register_lexicon(
    "calendar",
    "zh",
    {
        "event-title": (
            LexiconEntry("开会", "开会"),
            LexiconEntry("体检", "体检"),
            LexiconEntry("面试", "面试"),
        ),
        "start-time": (
            LexiconEntry("九点", "09:00"),
            LexiconEntry("十四点半", "14:30"),
        ),
        "end-time": (
            LexiconEntry("十点", "10:00"),
            LexiconEntry("十六点", "16:00"),
        ),
        "day": (
            LexiconEntry("今天", "今天"),
            LexiconEntry("明天", "明天"),
            LexiconEntry("周末", "周末"),
        ),
        "participant": (
            LexiconEntry("张伟", "张伟"),
            LexiconEntry("李娜", "李娜"),
        ),
    },
)


register_lexicon(
    "weather",
    "en",
    {
        "city": (
            LexiconEntry("Beijing", "北京"),
            LexiconEntry("Shanghai", "上海"),
        ),
        "day": (
            LexiconEntry("today", "今天"),
            LexiconEntry("tomorrow", "明天"),
        ),
        "clock": (
            LexiconEntry("seven thirty", "07:30"),
            LexiconEntry("nine", "09:00"),
        ),
        "metric": (
            LexiconEntry("temperature", "temperature"),
            LexiconEntry("rain", "rain"),
        ),
    },
)

register_lexicon(
    "music-control",
    "en",
    {
        "song": (
            LexiconEntry("Blue Porcelain", "青花瓷"),
            LexiconEntry("Rice Fields", "稻香"),
        ),
        "artist": (
            LexiconEntry("Zhou Jielun", "周杰伦"),
            LexiconEntry("Deng Ziqi", "邓紫棋"),
        ),
        "genre": (
            LexiconEntry("pop", "pop"),
            LexiconEntry("rock", "rock"),
        ),
        "volume-level": (
            LexiconEntry("thirty", 30),
            LexiconEntry("fifty", 50),
        ),
        "playlist": (
            LexiconEntry("my favorites", "我的收藏"),
            LexiconEntry("running mix", "跑步歌单"),
        ),
    },
)

register_lexicon(
    "navigation",
    "en",
    {
        "destination": (
            LexiconEntry("People's Square", "人民广场"),
            LexiconEntry("Hongqiao Airport", "虹桥机场"),
        ),
        "origin": (
            LexiconEntry("home", "家"),
            LexiconEntry("school", "学校"),
        ),
        "transport": (
            LexiconEntry("by car", "drive"),
            LexiconEntry("on foot", "walk"),
        ),
        "category": (
            LexiconEntry("restaurants", "food"),
            LexiconEntry("parking", "parking"),
        ),
    },
)

register_lexicon(
    "calendar",
    "en",
    {
        "event-title": (
            LexiconEntry("standup meeting", "开会"),
            LexiconEntry("checkup", "体检"),
        ),
        "start-time": (
            LexiconEntry("nine", "09:00"),
            LexiconEntry("two thirty", "14:30"),
        ),
        "end-time": (
            LexiconEntry("ten", "10:00"),
            LexiconEntry("four", "16:00"),
        ),
        "day": (
            LexiconEntry("today", "今天"),
            LexiconEntry("tomorrow", "明天"),
        ),
        "participant": (
            LexiconEntry("Zhang Wei", "张伟"),
            LexiconEntry("Li Na", "李娜"),
        ),
    },
)
