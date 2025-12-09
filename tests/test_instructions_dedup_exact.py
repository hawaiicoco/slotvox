"""Normalization, text hashing, and exact-duplicate groups."""

import re

import pytest

from slotvox.errors import ValidationError
from slotvox.instructions.dedup import (
    exact_duplicate_groups,
    normalize_text,
    sample_text,
    text_hash,
)
from slotvox.instructions.schema import AudioRef, InstructionSample, Turn

HASH = "b2" * 32


def make_sample(sample_id, text):
    return InstructionSample(
        sample_id=sample_id,
        domain="weather",
        language="zh",
        intent="query-weather",
        turns=(
            Turn("user", audio=AudioRef(sample_id, HASH, 300.0)),
            Turn("assistant", text=text),
        ),
    )


def test_normalization_collapses_surface_variants():
    assert normalize_text("Hello, World!") == "helloworld"
    assert normalize_text("hello   WORLD") == "helloworld"
    assert normalize_text("ＡＢＣ") == "abc"  # NFKC widens then casefolds
    assert normalize_text("意图：ＡＢＣ") == normalize_text("意图:abc")
    assert normalize_text("！！！   ") == ""
    with pytest.raises(ValidationError, match="string"):
        normalize_text(3)


def test_text_hash_stable_and_sensitive():
    assert re.fullmatch(r"[0-9a-f]{64}", text_hash("a"))
    assert text_hash("a b") == text_hash("A  B")
    assert text_hash("a") != text_hash("b")


def test_sample_text_excludes_audio_turns():
    sample = make_sample("s-one", "意图：x")
    assert sample_text(sample) == "意图：x"


def test_exact_groups_pair_variants_only():
    one = make_sample("s-one", "意图：x")
    two = make_sample("s-two", "意图: X")
    three = make_sample("s-three", "意图：y")
    assert exact_duplicate_groups([one, two, three]) == (("s-one", "s-two"),)
    assert exact_duplicate_groups([]) == ()
    assert exact_duplicate_groups([one]) == ()
    with pytest.raises(ValidationError, match="unique"):
        exact_duplicate_groups([one, one])
