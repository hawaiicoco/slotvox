"""All zh template tables validate against the built-in domains."""

import pytest

from slotvox.data.patterns import templates_for
from slotvox.errors import ValidationError
from slotvox.schema.builtins import builtin_domains


def test_every_zh_template_validates():
    for domain in builtin_domains():
        templates = templates_for(domain.domain, "zh")
        assert len(templates) >= 3
        for template in templates:
            template.validate_against(domain)
            assert template.language == "zh"


def test_every_intent_has_at_least_one_zh_template():
    for domain in builtin_domains():
        covered = {template.intent for template in templates_for(domain.domain, "zh")}
        assert covered == set(domain.intent_names)


def test_pattern_ids_unique_across_zh_tables():
    ids = [
        template.pattern_id
        for domain in builtin_domains()
        for template in templates_for(domain.domain, "zh")
    ]
    assert len(ids) == len(set(ids))


def test_first_weather_template_is_pinned():
    first = templates_for("weather", "zh")[0]
    assert first.pattern_id == "weather/query-weather/0"
    assert first.intent == "query-weather"
    assert len(first.tokens) == 7


def test_duplicate_registration_rejected():
    with pytest.raises(ValidationError, match="already registered"):
        from slotvox.data.patterns import register_templates

        register_templates("weather", "zh", [])
