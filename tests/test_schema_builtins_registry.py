"""Built-in navigation/calendar domains and the registry contracts."""

import pytest

from slotvox.errors import ValidationError
from slotvox.schema.builtins import (
    BUILTIN_DOMAIN_IDS,
    builtin_domain,
    builtin_domains,
    calendar_domain,
    navigation_domain,
)
from slotvox.schema.domains import DomainSpec


def test_navigation_and_calendar_invariants():
    nav = navigation_domain()
    assert nav.intent("find-route").required_slots == ("origin", "destination")
    assert nav.intent("search-nearby").slot_names == ("category",)
    cal = calendar_domain()
    assert cal.intent("create-event").required_slots == ("event-title", "start-time")
    cal.slot_spec("start-time").check_value("09:30")
    with pytest.raises(ValidationError):
        cal.slot_spec("start-time").check_value("9:30")


def test_registry_lists_all_four_ids_sorted():
    assert BUILTIN_DOMAIN_IDS == ("calendar", "music-control", "navigation", "weather")
    specs = builtin_domains()
    assert tuple(spec.domain for spec in specs) == BUILTIN_DOMAIN_IDS
    assert len({spec.domain_hash for spec in specs}) == 4


def test_builtin_domain_lookup_both_directions():
    assert builtin_domain("weather").domain == "weather"
    for bad in ("nope", "", None, 7):
        with pytest.raises(ValidationError, match="unknown built-in domain"):
            builtin_domain(bad)


def test_every_builtin_round_trips():
    for spec in builtin_domains():
        assert isinstance(spec, DomainSpec)
        assert DomainSpec.from_dict(spec.to_dict()) == spec
