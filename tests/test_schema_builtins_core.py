"""Built-in weather and music-control domain contracts."""

import pytest

from slotvox.errors import ValidationError
from slotvox.schema.builtins import music_control_domain, weather_domain
from slotvox.schema.naming import validate_id


def assert_domain_invariants(spec):
    validate_id(spec.domain, "domain")
    assert len(set(spec.slot_names)) == len(spec.slot_names)
    assert len(set(spec.intent_names)) == len(spec.intent_names)
    for name in spec.slot_names:
        validate_id(name, "slot")
    for intent in spec.intents:
        for ref in intent.slots:
            spec.slot_spec(ref.slot)  # resolves or raises


def test_weather_domain_invariants():
    spec = weather_domain()
    assert spec.domain == "weather"
    assert_domain_invariants(spec)
    assert spec.intent("query-weather").required_slots == ("city",)
    spec.slot_spec("city").check_value("北京")
    with pytest.raises(ValidationError):
        spec.slot_spec("city").check_value("成都")


def test_music_control_invariants():
    spec = music_control_domain()
    assert spec.domain == "music-control"
    assert_domain_invariants(spec)
    assert spec.intent("set-volume").required_slots == ("volume-level",)
    assert spec.intent("pause-music").slots == ()
    spec.slot_spec("volume-level").check_value(50)
    with pytest.raises(ValidationError):
        spec.slot_spec("volume-level").check_value(101)


def test_builtin_hashes_are_stable():
    assert weather_domain().domain_hash == weather_domain().domain_hash
    assert music_control_domain().domain_hash == music_control_domain().domain_hash
    assert weather_domain().domain_hash != music_control_domain().domain_hash
