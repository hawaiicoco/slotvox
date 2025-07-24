"""Golden pin for the schema package API surface."""

from slotvox import schema

GOLDEN_ALL = [
    "AnnotatedUtterance",
    "BUILTIN_DOMAIN_IDS",
    "DOMAIN_SCHEMA_ID",
    "DOMAIN_SCHEMA_VERSION",
    "DomainSpec",
    "ID_PATTERN",
    "IntentDefinition",
    "LANGUAGES",
    "MAX_ARTIFACT_BYTES",
    "MAX_ID_LENGTH",
    "SLOT_VALUE_TYPES",
    "SlotRef",
    "SlotTypeSpec",
    "builtin_domain",
    "builtin_domains",
    "calendar_domain",
    "iter_jsonl",
    "music_control_domain",
    "navigation_domain",
    "read_json",
    "read_jsonl",
    "validate_id",
    "weather_domain",
    "write_json_atomic",
    "write_jsonl",
]


def test_schema_api_surface_is_pinned():
    assert sorted(schema.__all__) == GOLDEN_ALL
    for name in GOLDEN_ALL:
        assert hasattr(schema, name)


def test_schema_objects_work_together():
    domain = schema.builtin_domain("weather")
    annotation = schema.AnnotatedUtterance(
        utterance_id="utt-demo",
        language="zh",
        tokens=("北京", "明天", "天气"),
        tags=("B-city", "B-day", "O"),
        intent="query-weather",
    )
    annotation.validate_against(domain)
    assert annotation.text == "北京明天天气"
    restored = schema.AnnotatedUtterance.from_dict(annotation.to_dict())
    restored.validate_against(schema.DomainSpec.from_dict(domain.to_dict()))
    assert restored == annotation
