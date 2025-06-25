"""IntentDefinition and SlotRef validation contracts."""

import pytest

from slotvox.errors import ValidationError
from slotvox.schema.intents import IntentDefinition, SlotRef


def test_intent_normalizes_slot_list_and_exposes_names():
    intent = IntentDefinition(
        "play-music", [SlotRef("song"), SlotRef("artist", optional=True)], "播放音乐"
    )
    assert intent.slot_names == ("song", "artist")
    assert intent.required_slots == ("song",)
    assert isinstance(intent.slots, tuple)


def test_duplicate_slot_refs_rejected():
    with pytest.raises(ValidationError, match="twice"):
        IntentDefinition("play-music", [SlotRef("song"), SlotRef("song", True)])


def test_empty_slots_allowed():
    assert IntentDefinition("stop").slots == ()
    assert IntentDefinition("stop").slot_names == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "Play"},
        {"name": ""},
        {"name": 7},
        {"slots": ["song"]},
        {"slots": "song"},
        {"slots": [SlotRef("song"), "artist"]},
        {"description": None},
    ],
)
def test_invalid_intents_rejected(kwargs):
    base = {"name": "play-music"}
    base.update(kwargs)
    with pytest.raises(ValidationError):
        IntentDefinition(**base)


def test_slot_ref_validation():
    with pytest.raises(ValidationError):
        SlotRef("Bad")
    with pytest.raises(ValidationError):
        SlotRef("ok", optional="yes")
    assert SlotRef("ok").optional is False


def test_slot_ref_round_trip_and_strict_keys():
    ref = SlotRef("song", True)
    assert SlotRef.from_dict(ref.to_dict()) == ref
    with pytest.raises(ValidationError, match="unknown keys"):
        SlotRef.from_dict({"slot": "song", "optional": False, "x": 1})
    with pytest.raises(ValidationError, match="missing keys"):
        SlotRef.from_dict({"slot": "song"})
