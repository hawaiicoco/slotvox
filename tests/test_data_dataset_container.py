"""GeneratedDataset container contracts (hand-assembled datasets)."""

import re
from dataclasses import replace

import pytest

from slotvox.config.generation import GenerationConfig
from slotvox.data.dataset import GeneratedDataset
from slotvox.data.factory import render_example
from slotvox.data.patterns import templates_for
from slotvox.errors import ValidationError
from slotvox.schema.builtins import weather_domain

CFG = GenerationConfig(domain="weather", seed=1, counts={"train": 1}, n_speakers=1)
TEMPLATE = templates_for("weather", "zh")[0]


def example(utterance_id="train-00000", seed=2):
    return render_example(
        weather_domain(),
        TEMPLATE,
        "zh",
        utterance_id=utterance_id,
        split="train",
        speaker_id=0,
        snr_db=30.0,
        seed=seed,
    )


def dataset(**overrides):
    data = {
        "domain_id": "weather",
        "language": "zh",
        "policy": "speaker",
        "seed": 1,
        "config": CFG,
        "examples": (example(),),
    }
    data.update(overrides)
    return GeneratedDataset(**data)


def test_valid_dataset_exposes_views():
    ds = dataset()
    assert ds.split_counts == {"train": 1, "dev": 0, "test": 0}
    assert ds.split_examples("train") == ds.examples
    assert ds.speakers_per_split()["train"] == (0,)
    assert ds.patterns_per_split()["train"] == (TEMPLATE.pattern_id,)


def test_provenance_envelope_and_hash():
    ds = dataset()
    prov = ds.provenance()
    assert prov["schema"] == "slotvox.dataset-provenance"
    assert prov["schema_version"] == 1
    assert prov["config"]["kind"] == "generation"
    assert re.fullmatch(r"[0-9a-f]{64}", ds.dataset_hash)
    assert dataset().dataset_hash == ds.dataset_hash


def test_count_mismatch_rejected():
    with pytest.raises(ValidationError, match="do not match"):
        dataset(examples=())


def test_duplicate_ids_rejected():
    two = GenerationConfig(domain="weather", seed=1, counts={"train": 2}, n_speakers=1)
    dup = example()
    with pytest.raises(ValidationError, match="duplicate utterance_id"):
        dataset(config=two, examples=(dup, dup))


def test_domain_mismatch_and_bad_enums_rejected():
    with pytest.raises(ValidationError, match="config domain"):
        dataset(domain_id="calendar")
    with pytest.raises(ValidationError, match="unknown built-in domain"):
        dataset(domain_id="nope")
    with pytest.raises(ValidationError):
        dataset(language="fr")
    with pytest.raises(ValidationError):
        dataset(policy="random")


def test_split_examples_strict():
    with pytest.raises(ValidationError, match="split must be"):
        dataset().split_examples("nope")


def test_replace_reruns_validation():
    with pytest.raises(ValidationError, match="do not match"):
        replace(dataset(), examples=())
