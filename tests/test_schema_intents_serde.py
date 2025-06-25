"""IntentDefinition serialization contracts, including a golden dict."""

import pytest

from slotvox.errors import ValidationError
from slotvox.schema.intents import IntentDefinition, SlotRef
from slotvox.util.jsoncanon import stable_hash


def intent():
    return IntentDefinition(
        "set-alarm",
        (SlotRef("time"), SlotRef("label", optional=True)),
        "设置闹钟",
    )


def test_round_trip_preserves_structure():
    assert IntentDefinition.from_dict(intent().to_dict()) == intent()


def test_to_dict_is_json_native():
    assert isinstance(stable_hash(intent().to_dict()), str)


def test_from_dict_strict_keys():
    payload = intent().to_dict()
    with pytest.raises(ValidationError, match="unknown keys"):
        IntentDefinition.from_dict({**payload, "priority": 1})
    missing = dict(payload)
    del missing["slots"]
    with pytest.raises(ValidationError, match="missing keys"):
        IntentDefinition.from_dict(missing)
    with pytest.raises(ValidationError, match="'slots' must be a list"):
        IntentDefinition.from_dict({**payload, "slots": {"slot": "time"}})


def test_nested_refs_reject_bad_payloads():
    payload = intent().to_dict()
    payload["slots"][0] = {"slot": "time"}
    with pytest.raises(ValidationError, match="missing keys"):
        IntentDefinition.from_dict(payload)


def test_golden_intent_dict():
    assert intent().to_dict() == {
        "name": "set-alarm",
        "slots": [
            {"slot": "time", "optional": False},
            {"slot": "label", "optional": True},
        ],
        "description": "设置闹钟",
    }
