"""DomainSpec validation contracts."""

import pytest

from slotvox.errors import ValidationError
from slotvox.schema.domains import DomainSpec
from slotvox.schema.intents import IntentDefinition, SlotRef
from slotvox.schema.slots import SlotTypeSpec


def make_domain(**overrides):
    data = {
        "domain": "weather",
        "slots": [SlotTypeSpec("city", "text"), SlotTypeSpec("clock", "time")],
        "intents": [IntentDefinition("query", [SlotRef("city"), SlotRef("clock", True)])],
    }
    data.update(overrides)
    return DomainSpec(**data)


def test_valid_domain_exposes_names_and_lookups():
    spec = make_domain()
    assert spec.slot_names == ("city", "clock")
    assert spec.intent_names == ("query",)
    assert spec.slot_spec("city").value_type == "text"
    assert spec.intent("query").required_slots == ("city",)


def test_unknown_lookups_rejected():
    with pytest.raises(ValidationError, match="has no slot"):
        make_domain().slot_spec("region")
    with pytest.raises(ValidationError, match="has no intent"):
        make_domain().intent("nope")


def test_lists_normalize_to_tuples():
    spec = make_domain()
    assert isinstance(spec.slots, tuple)
    assert isinstance(spec.intents, tuple)


def test_duplicate_slots_and_intents_rejected():
    with pytest.raises(ValidationError, match="duplicate slot"):
        make_domain(slots=[SlotTypeSpec("city", "text"), SlotTypeSpec("city", "time")])
    with pytest.raises(ValidationError, match="duplicate intent"):
        make_domain(intents=[IntentDefinition("query"), IntentDefinition("query")])


def test_unknown_slot_reference_rejected():
    with pytest.raises(ValidationError, match="unknown slot"):
        make_domain(intents=[IntentDefinition("query", [SlotRef("region")])])


def test_empty_intents_rejected():
    with pytest.raises(ValidationError, match="at least one intent"):
        make_domain(intents=[])


def test_bad_identity_and_version_rejected():
    with pytest.raises(ValidationError):
        make_domain(domain="Weather")
    with pytest.raises(ValidationError, match="unsupported schema version"):
        make_domain(schema_version=2)
