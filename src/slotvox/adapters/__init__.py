"""Inference-service adapters: protocol contracts and offline test doubles.

Everything here is offline by construction: the replay adapter serves
validated fixtures, the mock server binds to loopback only, and the HTTP
client refuses non-loopback URLs. Adapters validate protocol behavior,
not model quality. The torch-backed model adapter lives in
:mod:`slotvox.adapters.model` and requires the torch extra.
"""

from slotvox.adapters.http import RETRYABLE_STATUSES, LocalHttpAdapter
from slotvox.adapters.mock_server import MockInferServer
from slotvox.adapters.protocol import (
    MAX_REQUEST_SAMPLES,
    REQUEST_SCHEMA_ID,
    REQUEST_SCHEMA_VERSION,
    RESPONSE_SCHEMA_ID,
    RESPONSE_SCHEMA_VERSION,
    SUPPORTED_SAMPLE_RATES,
    InferRequest,
    InferResponse,
    request_schema,
    response_schema,
)
from slotvox.adapters.replay import (
    FIXTURE_SCHEMA_ID,
    FIXTURE_SCHEMA_VERSION,
    Fixture,
    ReplayAdapter,
    fixture_from_dict,
    load_fixtures,
    save_fixtures,
)
from slotvox.adapters.validator import SUPPORTED_KEYWORDS, validate

__all__ = [
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
