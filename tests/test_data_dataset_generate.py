"""generate_dataset contracts: counts, leakage, determinism, goldens.

Golden hashes were produced by executing generate_dataset with the pinned
config below; any drift must be a deliberate commit.
"""

from dataclasses import replace

import pytest

from slotvox.config.generation import GenerationConfig
from slotvox.data.dataset import generate_dataset
from slotvox.errors import ValidationError

CFG = GenerationConfig(
    domain="weather", seed=1234, counts={"train": 6, "dev": 3, "test": 2}, n_speakers=3
)

GOLDEN_SPEAKER_HASH = "57e9cf0c53dad2ef5d7168398d669e7328c87d68c569d2c06a0cf8eec0f0fcf9"
GOLDEN_PATTERN_HASH = "228428eeb523255d3c358eeba3eac9a4d86313f90ca3de20d0817416309c5ffd"


def test_counts_and_unique_ids():
    ds = generate_dataset(CFG)
    assert ds.split_counts == {"train": 6, "dev": 3, "test": 2}
    ids = [example.utterance_id for example in ds.examples]
    assert len(ids) == len(set(ids)) == 11
    assert ids[0] == "train-00000"


def test_golden_hashes():
    assert generate_dataset(CFG).dataset_hash == GOLDEN_SPEAKER_HASH
    assert generate_dataset(CFG, policy="pattern").dataset_hash == GOLDEN_PATTERN_HASH


def test_determinism_and_seed_sensitivity():
    assert generate_dataset(CFG).dataset_hash == generate_dataset(CFG).dataset_hash
    assert generate_dataset(CFG, seed=99).dataset_hash != generate_dataset(CFG).dataset_hash


def test_speaker_policy_is_leak_free():
    per = generate_dataset(CFG).speakers_per_split()
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        assert set(per[a]) & set(per[b]) == set()


def test_pattern_policy_is_leak_free():
    per = generate_dataset(CFG, policy="pattern").patterns_per_split()
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        assert set(per[a]) & set(per[b]) == set()


def test_container_rejects_tampered_examples():
    ds = generate_dataset(CFG)
    with pytest.raises(ValidationError, match="do not match"):
        replace(ds, examples=ds.examples[:1])


def test_invalid_arguments_rejected():
    with pytest.raises(ValidationError):
        generate_dataset(CFG, policy="random")
    with pytest.raises(ValidationError):
        generate_dataset(CFG, language="fr")
    with pytest.raises(ValidationError):
        generate_dataset("not-a-config")
    with pytest.raises(ValidationError):
        generate_dataset(CFG, seed=-3)


def test_english_generation_works():
    ds = generate_dataset(
        GenerationConfig(domain="music-control", seed=3, counts={"train": 4}, n_speakers=2),
        language="en",
    )
    assert ds.examples[0].annotation.language == "en"
    assert ds.split_counts["train"] == 4
