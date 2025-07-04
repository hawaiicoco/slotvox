"""DomainSpec serialization contracts, golden dict, provenance hash."""

import pytest

from slotvox.errors import SchemaError
from slotvox.schema.domains import DomainSpec
from slotvox.schema.intents import IntentDefinition, SlotRef
from slotvox.schema.slots import SlotTypeSpec


def make_domain():
    return DomainSpec(
        domain="weather",
        slots=(SlotTypeSpec("city", "text"), SlotTypeSpec("clock", "time")),
        intents=(IntentDefinition("query", (SlotRef("city"), SlotRef("clock", True))),),
    )


def test_round_trip_preserves_equality():
    spec = make_domain()
    assert DomainSpec.from_dict(spec.to_dict()) == spec


def test_golden_domain_dict():
    assert make_domain().to_dict() == {
        "schema": "slotvox.domain",
        "schema_version": 1,
        "domain": "weather",
        "description": "",
        "slots": [
            {
                "name": "city",
                "value_type": "text",
                "constraint": {"min_length": 1, "max_length": 512},
                "description": "",
            },
            {"name": "clock", "value_type": "time", "constraint": {}, "description": ""},
        ],
        "intents": [
            {
                "name": "query",
                "slots": [
                    {"slot": "city", "optional": False},
                    {"slot": "clock", "optional": True},
                ],
                "description": "",
            }
        ],
    }


def test_parse_rejects_bad_envelope():
    payload = make_domain().to_dict()
    with pytest.raises(SchemaError, match="unknown keys"):
        DomainSpec.from_dict({**payload, "extra": 1})
    with pytest.raises(SchemaError, match="missing keys"):
        DomainSpec.from_dict({k: v for k, v in payload.items() if k != "domain"})
    with pytest.raises(SchemaError, match="schema id"):
        DomainSpec.from_dict({**payload, "schema": "slotvox.other"})
    with pytest.raises(SchemaError, match="schema version"):
        DomainSpec.from_dict({**payload, "schema_version": 99})
    with pytest.raises(SchemaError, match="must be lists"):
        DomainSpec.from_dict({**payload, "slots": {}})


def test_hash_is_stable_and_sensitive():
    first, second = make_domain().domain_hash, make_domain().domain_hash
    assert first == second
    assert len(first) == 64
    changed = DomainSpec(
        domain="weather",
        slots=(SlotTypeSpec("city", "text", {"max_length": 64}), SlotTypeSpec("clock", "time")),
        intents=(IntentDefinition("query", (SlotRef("city"), SlotRef("clock", True))),),
    )
    assert changed.domain_hash != first
