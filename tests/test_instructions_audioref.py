"""AudioRef contracts: validation in both directions and serde."""

import pytest

from slotvox.errors import ValidationError
from slotvox.instructions.schema import MAX_DURATION_MS, AudioRef

HASH = "a" * 64


def test_valid_construction_and_widening():
    ref = AudioRef("weather-train-00000", HASH, 920)
    assert ref.duration_ms == 920.0
    assert ref.manifest_id == "weather-train-00000"


def test_roundtrip_and_strict_keys():
    ref = AudioRef("weather-train-00000", HASH, 920.0)
    payload = ref.to_dict()
    assert sorted(payload) == ["dataset_hash", "duration_ms", "manifest_id"]
    assert AudioRef.from_dict(payload) == ref
    with pytest.raises(ValidationError, match="unknown keys"):
        AudioRef.from_dict({**payload, "extra": 1})
    with pytest.raises(ValidationError, match="missing keys"):
        AudioRef.from_dict({"manifest_id": ref.manifest_id})
    with pytest.raises(ValidationError, match="dict"):
        AudioRef.from_dict("nope")


def test_validation_rejects_bad_values():
    with pytest.raises(ValidationError):
        AudioRef("Bad ID", HASH, 100.0)
    with pytest.raises(ValidationError, match="hex"):
        AudioRef("ok-id", "A" * 64, 100.0)
    with pytest.raises(ValidationError, match="hex"):
        AudioRef("ok-id", "a" * 63, 100.0)
    for bad in (0, -1.0, MAX_DURATION_MS + 1, "x"):
        with pytest.raises(ValidationError, match="duration"):
            AudioRef("ok-id", HASH, bad)
