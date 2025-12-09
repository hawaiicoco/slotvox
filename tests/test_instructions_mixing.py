"""Weighted sample mixing: determinism, quotas, and honest errors."""

import pytest

from slotvox.errors import ValidationError
from slotvox.instructions.mixing import mix_samples
from slotvox.instructions.schema import AudioRef, InstructionSample, Turn

HASH = "a1" * 32


def make_sample(sample_id, text):
    return InstructionSample(
        sample_id=sample_id,
        domain="weather",
        language="zh",
        intent="query-weather",
        turns=(
            Turn("user", audio=AudioRef(sample_id, HASH, 500.0)),
            Turn("assistant", text=text),
        ),
    )


def build_group(prefix, count):
    return tuple(
        make_sample(f"{prefix}-{index:03d}", f"text {prefix} {index}") for index in range(count)
    )


def test_mix_is_deterministic_and_sized():
    groups = {"a": build_group("a", 6), "b": build_group("b", 6)}
    mixed = mix_samples(groups, {"a": 1, "b": 1}, 6, seed=11)
    assert len(mixed) == 6
    assert mixed == mix_samples(groups, {"a": 1, "b": 1}, 6, seed=11)
    per_source = {}
    for sample in mixed:
        per_source[sample.sample_id.split("-")[0]] = (
            per_source.get(sample.sample_id.split("-")[0], 0) + 1
        )
    assert per_source == {"a": 3, "b": 3}


def test_input_order_does_not_matter():
    forward = {"a": build_group("a", 6), "b": build_group("b", 6)}
    reversed_groups = {"b": forward["b"][::-1], "a": forward["a"][::-1]}
    assert mix_samples(forward, {"a": 1, "b": 1}, 6, 11) == mix_samples(
        reversed_groups, {"a": 1, "b": 1}, 6, 11
    )


def test_seed_changes_order():
    groups = {"a": build_group("a", 6)}
    one = mix_samples(groups, {"a": 1}, 6, seed=1)
    two = mix_samples(groups, {"a": 1}, 6, seed=2)
    assert sorted(sample.sample_id for sample in one) == sorted(sample.sample_id for sample in two)
    assert one != two


def test_availability_and_duplicate_ids():
    with pytest.raises(ValidationError, match="quota"):
        mix_samples({"a": build_group("a", 2)}, {"a": 1}, 5, seed=1)
    with pytest.raises(ValidationError, match="duplicate"):
        mix_samples(
            {"a": build_group("a", 3), "b": build_group("a", 3)}, {"a": 1, "b": 1}, 6, seed=1
        )


def test_source_key_mismatch_both_directions():
    groups = {"a": build_group("a", 3)}
    with pytest.raises(ValidationError, match="unknown"):
        mix_samples({**groups, "b": build_group("b", 3)}, {"a": 1}, 3, seed=1)
    with pytest.raises(ValidationError, match="missing"):
        mix_samples(groups, {"a": 1, "b": 1}, 3, seed=1)


def test_bad_inputs():
    with pytest.raises(ValidationError, match="InstructionSample"):
        mix_samples({"a": ("x",)}, {"a": 1}, 1, seed=1)
    with pytest.raises(ValidationError, match="list/tuple"):
        mix_samples({"a": "nope"}, {"a": 1}, 1, seed=1)
    with pytest.raises(ValidationError, match="seed"):
        mix_samples({"a": build_group("a", 1)}, {"a": 1}, 1, seed=-5)
