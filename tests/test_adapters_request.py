"""InferRequest contracts: bounds, enums, and strict serde."""

import pytest

from slotvox.adapters.protocol import (
    MAX_REQUEST_SAMPLES,
    REQUEST_SCHEMA_ID,
    REQUEST_SCHEMA_VERSION,
    InferRequest,
)
from slotvox.errors import ValidationError


def test_valid_request_and_widening():
    request = InferRequest("req-1", [0, 1.5, -2], 16000)
    assert request.samples == (0.0, 1.5, -2.0)
    assert request.to_dict()["samples"] == [0.0, 1.5, -2.0]


def test_roundtrip_and_envelope():
    request = InferRequest("req-1", (0.1,), 8000)
    payload = request.to_dict()
    assert payload["schema"] == REQUEST_SCHEMA_ID
    assert payload["schema_version"] == REQUEST_SCHEMA_VERSION
    assert InferRequest.from_dict(payload) == request
    with pytest.raises(ValidationError, match="unknown keys"):
        InferRequest.from_dict({**payload, "extra": 1})
    with pytest.raises(ValidationError, match="missing keys"):
        InferRequest.from_dict({"schema": REQUEST_SCHEMA_ID})
    with pytest.raises(ValidationError, match="schema"):
        InferRequest.from_dict({**payload, "schema": "other"})
    with pytest.raises(ValidationError, match="version"):
        InferRequest.from_dict({**payload, "schema_version": 9})
    with pytest.raises(ValidationError, match="list"):
        InferRequest.from_dict({**payload, "samples": (0.1,)})


def test_validation_both_directions():
    with pytest.raises(ValidationError, match="non-empty"):
        InferRequest("", (0.1,), 16000)
    with pytest.raises(ValidationError, match="whitespace"):
        InferRequest("a b", (0.1,), 16000)
    with pytest.raises(ValidationError, match="non-empty"):
        InferRequest("r", (), 16000)
    with pytest.raises(ValidationError, match="finite"):
        InferRequest("r", (float("nan"),), 16000)
    with pytest.raises(ValidationError, match="numbers"):
        InferRequest("r", ("x",), 16000)
    with pytest.raises(ValidationError, match="sample_rate"):
        InferRequest("r", (0.1,), 1234)


def test_sample_bound():
    with pytest.raises(ValidationError, match="bound"):
        InferRequest("r", [0.0] * (MAX_REQUEST_SAMPLES + 1), 16000)
