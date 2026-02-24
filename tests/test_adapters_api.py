"""Golden pin for the adapters package API surface."""

import slotvox.adapters as adapters

GOLDEN_ALL = [
    "FIXTURE_SCHEMA_ID",
    "FIXTURE_SCHEMA_VERSION",
    "Fixture",
    "InferRequest",
    "InferResponse",
    "LocalHttpAdapter",
    "MAX_REQUEST_SAMPLES",
    "MockInferServer",
    "REQUEST_SCHEMA_ID",
    "REQUEST_SCHEMA_VERSION",
    "RESPONSE_SCHEMA_ID",
    "RESPONSE_SCHEMA_VERSION",
    "RETRYABLE_STATUSES",
    "ReplayAdapter",
    "SUPPORTED_KEYWORDS",
    "SUPPORTED_SAMPLE_RATES",
    "fixture_from_dict",
    "load_fixtures",
    "request_schema",
    "response_schema",
    "save_fixtures",
    "validate",
]


def test_api_surface_is_pinned():
    assert sorted(adapters.__all__) == GOLDEN_ALL
    for name in GOLDEN_ALL:
        assert hasattr(adapters, name)


def test_protocol_constants():
    assert adapters.REQUEST_SCHEMA_ID == "slotvox.infer-request"
    assert adapters.RESPONSE_SCHEMA_ID == "slotvox.infer-response"
    assert adapters.FIXTURE_SCHEMA_ID == "slotvox.adapter-fixtures"
    assert adapters.RETRYABLE_STATUSES == (429, 500, 502, 503, 504)
    assert 16000 in adapters.SUPPORTED_SAMPLE_RATES
