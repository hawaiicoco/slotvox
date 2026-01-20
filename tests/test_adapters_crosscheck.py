"""Protocol payloads vs protocol schemas: the validator cross-check."""

import pytest

from slotvox.adapters.protocol import InferRequest, InferResponse, request_schema, response_schema
from slotvox.adapters.validator import validate
from slotvox.errors import AdapterError

HASH = "ef" * 32


def request():
    return InferRequest("req-1", (0.1, 0.2), 16000)


def response():
    return InferResponse("req-1", "query-weather", 0.8, ("O", "B-city"), HASH)


def test_valid_payloads_pass():
    validate(request().to_dict(), request_schema())
    validate(response().to_dict(), response_schema())


def test_request_mutations_rejected():
    payload = request().to_dict()
    mutations = [
        {**payload, "schema": "other"},
        {**payload, "extra": 1},
        {**payload, "samples": []},
        {**payload, "samples": ["x"]},
        {**payload, "sample_rate": 1234},
        {key: value for key, value in payload.items() if key != "request_id"},
    ]
    for mutation in mutations:
        with pytest.raises(AdapterError):
            validate(mutation, request_schema())


def test_response_mutations_rejected():
    payload = response().to_dict()
    mutations = [
        {**payload, "posterior": 1.7},
        {**payload, "posterior": -0.2},
        {**payload, "model_hash": "XYZ"},
        {**payload, "frame_tags": []},
        {**payload, "intent": ""},
        {**payload, "schema_version": 12},
    ]
    for mutation in mutations:
        with pytest.raises(AdapterError):
            validate(mutation, response_schema())
