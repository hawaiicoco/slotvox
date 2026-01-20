"""Shared adapter fixtures (synthetic, protocol-shaped)."""

from slotvox.adapters.protocol import InferRequest, InferResponse
from slotvox.adapters.replay import Fixture

HASH = "33" * 32


def make_request(request_id="req-1"):
    return InferRequest(request_id, (0.1, 0.2, 0.3), 16000)


def make_response(request_id="req-1"):
    return InferResponse(request_id, "query-weather", 0.9, ("O", "B-city", "O"), HASH)


def make_fixture(request_id="req-1"):
    return Fixture(make_request(request_id), make_response(request_id))
