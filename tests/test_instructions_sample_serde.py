"""Envelope goldens, roundtrips, and hash sensitivity for samples."""

import re

import pytest

from slotvox.errors import ValidationError
from slotvox.instructions.schema import (
    INSTRUCTION_SCHEMA_ID,
    INSTRUCTION_SCHEMA_VERSION,
    AudioRef,
    InstructionSample,
    Turn,
)

HASH = "d" * 64
GOLDEN_KEYS = [
    "domain",
    "intent",
    "language",
    "sample_id",
    "schema",
    "schema_version",
    "tags",
    "turns",
]


def build(**overrides):
    data = {
        "sample_id": "weather-train-00000",
        "domain": "weather",
        "language": "zh",
        "intent": "query-weather",
        "turns": (
            Turn("system", text="prompt"),
            Turn("user", audio=AudioRef("weather-train-00000", HASH, 500.0)),
            Turn("assistant", text="意图：query-weather；槽位：无"),
        ),
        "tags": ("synthetic",),
    }
    data.update(overrides)
    return InstructionSample(**data)


def test_envelope_golden_and_roundtrip():
    payload = build().to_dict()
    assert sorted(payload) == GOLDEN_KEYS
    assert payload["schema"] == INSTRUCTION_SCHEMA_ID
    assert payload["schema_version"] == INSTRUCTION_SCHEMA_VERSION
    assert sorted(payload["turns"][0]) == ["audio", "role", "text"]
    assert InstructionSample.from_dict(payload) == build()


def test_hash_stable_and_sensitive():
    assert re.fullmatch(r"[0-9a-f]{64}", build().sample_hash)
    assert build().sample_hash == build().sample_hash
    assert build(tags=("synthetic", "clean")).sample_hash != build().sample_hash


def test_from_dict_strict():
    payload = build().to_dict()
    with pytest.raises(ValidationError, match="unknown keys"):
        InstructionSample.from_dict({**payload, "extra": 1})
    with pytest.raises(ValidationError, match="missing keys"):
        InstructionSample.from_dict({key: payload[key] for key in GOLDEN_KEYS[:5]})
    with pytest.raises(ValidationError, match="schema"):
        InstructionSample.from_dict({**payload, "schema": "other"})
    with pytest.raises(ValidationError, match="version"):
        InstructionSample.from_dict({**payload, "schema_version": 99})
    with pytest.raises(ValidationError, match="list"):
        InstructionSample.from_dict({**payload, "turns": "nope"})
