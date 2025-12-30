"""InferResponse contracts: BIO structure, posterior bounds, serde."""

import pytest

from slotvox.adapters.protocol import RESPONSE_SCHEMA_ID, InferResponse
from slotvox.errors import ValidationError

HASH = "cd" * 32


def test_valid_response_and_roundtrip():
    response = InferResponse("req-1", "query-weather", 1, ["O", "B-city"], HASH)
    assert response.posterior == 1.0
    assert response.frame_tags == ("O", "B-city")
    payload = response.to_dict()
    assert payload["schema"] == RESPONSE_SCHEMA_ID
    assert InferResponse.from_dict(payload) == response


def test_posterior_bounds():
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        InferResponse("req-1", "a", 1.2, ("O",), HASH)
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        InferResponse("req-1", "a", -0.1, ("O",), HASH)
    with pytest.raises(ValidationError, match="number"):
        InferResponse("req-1", "a", "high", ("O",), HASH)


def test_frame_tags_must_be_valid_bio():
    with pytest.raises(ValidationError):
        InferResponse("req-1", "a", 0.5, (), HASH)  # empty
    with pytest.raises(ValidationError):
        InferResponse("req-1", "a", 0.5, ("I-city",), HASH)  # stray continuation
    with pytest.raises(ValidationError):
        InferResponse("req-1", "a", 0.5, ("O", "nope"), HASH)  # not a tag


def test_strict_serde_and_hash_guard():
    payload = InferResponse("req-1", "a", 0.5, ("O",), HASH).to_dict()
    with pytest.raises(ValidationError, match="unknown keys"):
        InferResponse.from_dict({**payload, "extra": 1})
    with pytest.raises(ValidationError, match="schema"):
        InferResponse.from_dict({**payload, "schema": "other"})
    with pytest.raises(ValidationError, match="hex"):
        InferResponse("req-1", "a", 0.5, ("O",), "A" * 64)
    with pytest.raises(ValidationError, match="intent"):
        InferResponse("req-1", " ", 0.5, ("O",), HASH)
