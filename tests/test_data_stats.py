"""summarize contracts with golden values from the pinned dataset."""

import pytest

from slotvox.config.generation import GenerationConfig
from slotvox.data.dataset import generate_dataset
from slotvox.data.stats import summarize
from slotvox.errors import ValidationError
from slotvox.util.jsoncanon import canonical_dumps

CFG = GenerationConfig(
    domain="weather", seed=1234, counts={"train": 6, "dev": 3, "test": 2}, n_speakers=3
)
DS = generate_dataset(CFG)

GOLDEN_KEYS = [
    "by_intent",
    "counts",
    "dataset_hash",
    "domain_id",
    "duration_s",
    "language",
    "patterns",
    "policy",
    "schema",
    "schema_version",
    "seed",
    "slot_mentions",
    "snr_levels",
    "speakers",
    "token_length",
]


def test_envelope_and_schema():
    stats = summarize(DS)
    assert stats["schema"] == "slotvox.dataset-stats"
    assert stats["schema_version"] == 1
    assert stats["dataset_hash"] == DS.dataset_hash
    assert sorted(stats) == GOLDEN_KEYS


def test_golden_statistics():
    stats = summarize(DS)
    assert stats["counts"] == {"train": 6, "dev": 3, "test": 2}
    assert stats["speakers"] == {"train": [1], "dev": [2], "test": [0]}
    assert stats["token_length"] == {"min": 6, "max": 9, "mean": 7.4545}
    assert stats["duration_s"] == {"total": 8.42, "mean": 0.765455}
    assert stats["by_intent"]["train"] == {
        "query-forecast": 2,
        "query-weather": 2,
        "weather-alert": 2,
    }
    assert stats["slot_mentions"] == {"city": 9, "day": 6}


def test_summary_is_canonical_json_ready():
    text = canonical_dumps(summarize(DS))
    assert isinstance(text, str)
    assert "dataset_hash" in text


def test_summarize_rejects_other_types():
    with pytest.raises(ValidationError, match="GeneratedDataset"):
        summarize("not-a-dataset")
