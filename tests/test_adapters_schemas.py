"""Protocol JSON-schema goldens (content pins, cross-checked separately)."""

import json

from slotvox.adapters.protocol import (
    MAX_REQUEST_SAMPLES,
    REQUEST_SCHEMA_ID,
    RESPONSE_SCHEMA_ID,
    SUPPORTED_SAMPLE_RATES,
    request_schema,
    response_schema,
)
from slotvox.util.jsoncanon import stable_hash


def test_request_schema_golden():
    schema = request_schema()
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == [
        "schema",
        "schema_version",
        "request_id",
        "samples",
        "sample_rate",
    ]
    assert schema["properties"]["schema"]["enum"] == [REQUEST_SCHEMA_ID]
    assert schema["properties"]["sample_rate"]["enum"] == list(SUPPORTED_SAMPLE_RATES)
    assert schema["properties"]["samples"]["maxItems"] == MAX_REQUEST_SAMPLES
    assert schema["properties"]["samples"]["items"] == {"type": "number"}


def test_response_schema_golden():
    schema = response_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema"]["enum"] == [RESPONSE_SCHEMA_ID]
    assert schema["properties"]["posterior"] == {"type": "number", "minimum": 0.0, "maximum": 1.0}
    assert schema["properties"]["model_hash"] == {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    assert schema["properties"]["frame_tags"]["minItems"] == 1


def test_schemas_are_canonical_json_ready():
    assert stable_hash(request_schema()) == stable_hash(request_schema())
    assert stable_hash(request_schema()) != stable_hash(response_schema())
    json.dumps(request_schema())
    json.dumps(response_schema())
