"""render_sample: references, determinism, and rejection contracts."""

import pytest

from slotvox.config import GenerationConfig
from slotvox.data.dataset import generate_dataset
from slotvox.errors import ValidationError
from slotvox.instructions.schema import InstructionSample
from slotvox.instructions.templates import render_sample

HASH = "e" * 64


def dataset():
    return generate_dataset(
        GenerationConfig(domain="weather", seed=23, counts={"train": 3}, n_speakers=1)
    )


def test_render_all_examples_valid_and_referenced():
    data = dataset()
    samples = [
        render_sample(example, domain="weather", dataset_hash=data.dataset_hash)
        for example in data.examples
    ]
    assert len(samples) == 3
    for example, rendered in zip(data.examples, samples, strict=True):
        assert isinstance(rendered, InstructionSample)
        audio = rendered.turns[1].audio
        assert audio is not None
        assert audio.manifest_id == f"weather-{example.utterance_id}"
        assert audio.dataset_hash == data.dataset_hash
        expected_ms = example.utterance.n_samples / example.utterance.sample_rate * 1000.0
        assert audio.duration_ms == pytest.approx(expected_ms)
        assert rendered.sample_id == audio.manifest_id
        assert rendered.tags == ("synthetic",)
        assert rendered.language == example.annotation.language
        assert example.annotation.intent in rendered.turns[2].text


def test_render_includes_slot_surface_forms():
    data = dataset()
    for example in data.examples:
        rendered = render_sample(example, domain="weather", dataset_hash=data.dataset_hash)
        text = rendered.turns[2].text
        for values in example.slot_values.values():
            for value in values:
                assert value in text


def test_render_deterministic():
    data = dataset()
    example = data.examples[0]
    first = render_sample(example, domain="weather", dataset_hash=data.dataset_hash)
    second = render_sample(example, domain="weather", dataset_hash=data.dataset_hash)
    assert first == second
    assert first.sample_hash == second.sample_hash


def test_render_rejections():
    data = dataset()
    example = data.examples[0]
    with pytest.raises(ValidationError, match="GeneratedExample"):
        render_sample("nope", domain="weather", dataset_hash=HASH)
    with pytest.raises(ValidationError, match="unknown built-in domain"):
        render_sample(example, domain="nope", dataset_hash=HASH)
    with pytest.raises(ValidationError, match="hex"):
        render_sample(example, domain="weather", dataset_hash="x")
    with pytest.raises(ValidationError, match="unknown slots"):
        render_sample(example, domain="music-control", dataset_hash=HASH)
