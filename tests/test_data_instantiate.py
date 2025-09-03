"""instantiate_template contracts: alignment, determinism, recovery."""

import pytest

from slotvox.data.factory import instantiate_template
from slotvox.data.patterns import templates_for
from slotvox.errors import ValidationError
from slotvox.tagging.bio import is_valid_sequence
from slotvox.util.jsoncanon import canonical_dumps
from slotvox.util.seed import make_rng

TEMPLATE = templates_for("weather", "zh")[0]  # [city][day]天气怎么样


def test_tokens_and_tags_align_and_are_valid():
    tokens, tags, values = instantiate_template("weather", "zh", TEMPLATE, make_rng(3))
    assert len(tokens) == len(tags)
    assert is_valid_sequence(tags)
    assert set(values) <= {"city", "day"}
    assert set(values) >= {"city"}  # required slot always present


def test_slot_surface_recoverable_from_tokens():
    tokens, tags, values = instantiate_template("weather", "zh", TEMPLATE, make_rng(4))
    span = [tok for tok, tag in zip(tokens, tags, strict=True) if tag.endswith("city")]
    assert "".join(span) == values["city"][0]


def test_literal_tail_is_outside():
    tokens, tags, _ = instantiate_template("weather", "zh", TEMPLATE, make_rng(6))
    assert tags[-5:] == ("O",) * 5
    assert tokens[-5:] == ("天", "气", "怎", "么", "样")


def test_deterministic_per_seed_and_varies_across_seeds():
    a = instantiate_template("weather", "zh", TEMPLATE, make_rng(5))
    b = instantiate_template("weather", "zh", TEMPLATE, make_rng(5))
    assert a == b
    outcomes = {
        canonical_dumps(instantiate_template("weather", "zh", TEMPLATE, make_rng(s))[2])
        for s in range(12)
    }
    assert len(outcomes) >= 2


def test_invalid_inputs_rejected():
    with pytest.raises(ValidationError):
        instantiate_template("weather", "zh", "not-a-template", make_rng(1))
    with pytest.raises(ValidationError):
        instantiate_template("weather", "zh", TEMPLATE, 42)
    with pytest.raises(ValidationError):
        instantiate_template("", "zh", TEMPLATE, make_rng(1))
