"""All en template tables validate; registry lookups are strict."""

import pytest

from slotvox.data.patterns import registered_templates, templates_for
from slotvox.errors import ValidationError
from slotvox.schema.builtins import builtin_domains


def test_every_en_template_validates():
    for domain in builtin_domains():
        templates = templates_for(domain.domain, "en")
        assert len(templates) >= 3
        for template in templates:
            template.validate_against(domain)
            assert template.language == "en"


def test_every_intent_has_at_least_one_en_template():
    for domain in builtin_domains():
        covered = {template.intent for template in templates_for(domain.domain, "en")}
        assert covered == set(domain.intent_names)


def test_registry_lists_all_domain_language_pairs():
    keys = set(registered_templates())
    for domain in builtin_domains():
        assert (domain.domain, "zh") in keys
        assert (domain.domain, "en") in keys


def test_strict_lookup_rejections():
    with pytest.raises(ValidationError, match="no templates registered"):
        templates_for("nope", "zh")
    with pytest.raises(ValidationError, match="no templates registered"):
        templates_for("weather", "fr")


def test_first_en_calendar_template_is_pinned():
    first = templates_for("calendar", "en")[0]
    assert first.pattern_id == "calendar/create-event/0"
    assert [token.slot for token in first.tokens if token.is_slot] == [
        "event-title",
        "start-time",
    ]
