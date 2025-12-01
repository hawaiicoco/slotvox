"""Template variable extraction and strict rendering."""

import pytest

from slotvox.errors import ValidationError
from slotvox.instructions.templates import TurnTemplate, template_variables


def test_variable_extraction_sorted_unique():
    assert template_variables("hear {b} and {a}, again {b}") == ("a", "b")
    assert template_variables("no variables here") == ()


def test_malformed_braces_rejected():
    with pytest.raises(ValidationError, match="braces"):
        template_variables("{Bad}")
    with pytest.raises(ValidationError, match="braces"):
        template_variables("stray } brace")
    with pytest.raises(ValidationError, match="braces"):
        TurnTemplate("t", "system", "{a} }")
    with pytest.raises(ValidationError, match="non-blank"):
        TurnTemplate("t", "system", "   ")


def test_render_strict_both_directions():
    template = TurnTemplate("t", "assistant", "意图：{intent}")
    assert template.render({"intent": "query-weather"}).text == "意图：query-weather"
    with pytest.raises(ValidationError, match="missing variables"):
        template.render({})
    with pytest.raises(ValidationError, match="unknown variables"):
        template.render({"intent": "x", "slots": "y"})
    with pytest.raises(ValidationError, match="non-blank"):
        template.render({"intent": ""})
    with pytest.raises(ValidationError, match="braces"):
        template.render({"intent": "{x}"})
    with pytest.raises(ValidationError, match="mapping"):
        template.render([("intent", "x")])


def test_template_construction_validation():
    with pytest.raises(ValidationError):
        TurnTemplate("Bad Id", "system", "x")
    with pytest.raises(ValidationError, match="role"):
        TurnTemplate("t", "tool", "x")
    assert TurnTemplate("t", "system", "{a}{b}").variables == ("a", "b")
