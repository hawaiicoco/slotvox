"""Inference-service adapters: protocol contracts and offline test doubles.

Everything here is offline by construction: the replay adapter serves
validated fixtures, the mock server binds to loopback only, and the HTTP
client refuses non-loopback URLs. Adapters validate protocol behavior,
not model quality. The torch-backed model adapter is exported lazily
(:class:`JointAdapter`) and requires the torch extra.
"""

import importlib
from typing import Any

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

_TORCH_MODULES = {"JointAdapter": "slotvox.adapters.model"}

__all__ = [
    "FIXTURE_SCHEMA_ID",
    "FIXTURE_SCHEMA_VERSION",
    "Fixture",
    "InferRequest",
    "InferResponse",
    "JointAdapter",
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


def __getattr__(name: str) -> Any:
    """Lazily resolve torch-backed exports with a helpful error when absent."""
    module_name = _TORCH_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'slotvox.adapters' has no attribute {name!r}")
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"slotvox.adapters.{name} requires the torch extra: "
            "pip install 'slotvox[torch]' "
            "(CPU wheels: --index-url https://download.pytorch.org/whl/cpu)"
        ) from exc
    return getattr(importlib.import_module(module_name), name)
