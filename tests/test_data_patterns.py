"""Pattern template machinery contracts."""

import pytest

from slotvox.data.patterns import (
    PatternTemplate,
    TokenPlan,
    register_templates,
    templates_for,
)
from slotvox.errors import ValidationError
from slotvox.schema.builtins import weather_domain


def lit(*texts):
    return tuple(TokenPlan(text=t) for t in texts)


def test_token_plan_requires_exactly_one_field():
    assert TokenPlan(text="天").is_slot is False
    assert TokenPlan(slot="city").is_slot is True
    with pytest.raises(ValidationError):
        TokenPlan()
    with pytest.raises(ValidationError):
        TokenPlan(text="天", slot="city")
    with pytest.raises(ValidationError):
        TokenPlan(text="two words")
    with pytest.raises(ValidationError):
        TokenPlan(slot="Bad")


def test_template_id_format_enforced():
    for bad in ("query-weather/0", "weather/query-weather/x", "a/b/c/d", ""):
        with pytest.raises(ValidationError):
            PatternTemplate(pattern_id=bad, intent="query-weather", tokens=lit("天"))
    with pytest.raises(ValidationError):
        PatternTemplate(
            pattern_id="weather/other-intent/0", intent="query-weather", tokens=lit("天")
        )


def test_adjacent_same_slot_placeholders_rejected():
    with pytest.raises(ValidationError, match="adjacent"):
        PatternTemplate(
            pattern_id="weather/query-weather/0",
            intent="query-weather",
            tokens=(TokenPlan(slot="city"), TokenPlan(slot="city")),
        )


def test_validate_against_domain_both_directions():
    domain = weather_domain()
    good = PatternTemplate(
        pattern_id="weather/query-weather/9",
        intent="query-weather",
        tokens=(TokenPlan(slot="city"),) + lit("天", "气"),
    )
    good.validate_against(domain)
    unknown_slot = PatternTemplate(
        pattern_id="weather/query-weather/9",
        intent="query-weather",
        tokens=(TokenPlan(slot="song"),),
    )
    with pytest.raises(ValidationError, match="not declared"):
        unknown_slot.validate_against(domain)
    missing_required = PatternTemplate(
        pattern_id="weather/query-weather/9",
        intent="query-weather",
        tokens=lit("天", "气"),
    )
    with pytest.raises(ValidationError, match="omits required"):
        missing_required.validate_against(domain)
    wrong_domain = PatternTemplate(
        pattern_id="calendar/query-weather/9",
        intent="query-weather",
        tokens=(TokenPlan(slot="city"),),
    )
    with pytest.raises(ValidationError, match="names domain"):
        wrong_domain.validate_against(domain)


def test_registry_strict_lookups():
    with pytest.raises(ValidationError, match="unknown built-in domain"):
        register_templates("nope", "zh", [])
    with pytest.raises(ValidationError, match="language must be"):
        register_templates("weather", "fr", [])
    with pytest.raises(ValidationError, match="no templates registered"):
        templates_for("weather", "fr")
    with pytest.raises(ValidationError):
        register_templates("weather", "zh", [])  # empty or duplicate: always rejected
