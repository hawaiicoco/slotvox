"""MockInferServer: wire shapes, scripted failures, and accounting."""

import json
import urllib.error
import urllib.request

import pytest

from slotvox.adapters.mock_server import MockInferServer
from slotvox.adapters.protocol import InferRequest, response_schema
from slotvox.adapters.replay import ReplayAdapter
from slotvox.adapters.validator import validate
from slotvox.errors import AdapterError
from tests.test_adapters_helpers import make_fixture, make_request


def post(url, payload, path="/v1/infer"):
    body = json.dumps(payload).encode("utf-8")
    raw = urllib.request.Request(
        url + path, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(raw, timeout=5) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def server():
    return MockInferServer(ReplayAdapter((make_fixture(),)))


def test_roundtrip_matches_response_schema():
    with server() as mock:
        status, payload = post(mock.base_url, make_request().to_dict())
    assert status == 200
    validate(payload, response_schema())
    assert payload["request_id"] == "req-1"


def test_bad_requests_get_400_never_a_decision():
    with server() as mock:
        status, body = post(mock.base_url, {"nope": 1})
        assert status == 400 and "error" in body
        broken = make_request().to_dict()
        broken["samples"] = []
        status, body = post(mock.base_url, broken)
        assert status == 400 and "error" in body


def test_unknown_id_404_and_route_shapes():
    with server() as mock:
        status, _ = post(mock.base_url, InferRequest("req-404", (0.1,), 16000).to_dict())
        assert status == 404
        status, _ = post(mock.base_url, make_request().to_dict(), path="/nope")
        assert status == 404
        with pytest.raises(urllib.error.HTTPError) as info:
            urllib.request.urlopen(mock.base_url + "/v1/infer", timeout=5)
        assert info.value.code == 405


def test_scripted_failures_and_counter():
    with server() as mock:
        mock.fail_next(503)
        status, body = post(mock.base_url, make_request().to_dict())
        assert status == 503
        assert body == {"error": "scripted failure"}
        status, _ = post(mock.base_url, make_request().to_dict())
        assert status == 200
        assert mock.requests_served == 2
        with pytest.raises(AdapterError, match=r"\[400, 599\]"):
            mock.fail_next(200)
        with pytest.raises(AdapterError, match="times"):
            mock.fail_next(500, times=0)
        with pytest.raises(AdapterError, match="ReplayAdapter"):
            MockInferServer("nope")
