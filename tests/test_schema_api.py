"""Golden pin for the schema package API surface (part 1)."""

from slotvox import schema

GOLDEN_ALL = [
    "ID_PATTERN",
    "IntentDefinition",
    "MAX_ID_LENGTH",
    "SLOT_VALUE_TYPES",
    "SlotRef",
    "SlotTypeSpec",
    "validate_id",
]


def test_schema_api_surface_is_pinned():
    assert sorted(schema.__all__) == GOLDEN_ALL
    for name in GOLDEN_ALL:
        assert hasattr(schema, name)


def test_cross_object_contract():
    spec = schema.SlotTypeSpec("time", "time")
    spec.check_value("07:30")
    intent = schema.IntentDefinition("set-alarm", [schema.SlotRef(spec.name)])
    assert intent.required_slots == (spec.name,)
    assert schema.validate_id(spec.name, "slot") == intent.slot_names[0]
