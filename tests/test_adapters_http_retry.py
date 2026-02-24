"""LocalHttpAdapter retry/timeout behavior against the loopback mock."""

import pytest

from slotvox.adapters.http import LocalHttpAdapter
from slotvox.adapters.mock_server import MockInferServer
from slotvox.adapters.replay import ReplayAdapter
from slotvox.errors import AdapterError
from tests.test_adapters_helpers import make_fixture, make_request


def server():
    return MockInferServer(ReplayAdapter((make_fixture(),)))


def test_roundtrip():
    with server() as mock:
        client = LocalHttpAdapter(mock.base_url)
        response = client.infer(make_request())
    assert response.intent == "query-weather"
    assert response.frame_tags == ("O", "B-city", "O")


def test_retries_then_succeeds():
    with server() as mock:
        mock.fail_next(503, times=2)
        client = LocalHttpAdapter(mock.base_url, retries=2, retry_delay_s=0.0)
        response = client.infer(make_request())
        assert response.request_id == "req-1"
        assert mock.requests_served == 3


def test_retry_budget_exhausted():
    with server() as mock:
        mock.fail_next(500, times=5)
        client = LocalHttpAdapter(mock.base_url, retries=1, retry_delay_s=0.0)
        with pytest.raises(AdapterError, match="HTTP 500"):
            client.infer(make_request())
        assert mock.requests_served == 2


def test_non_retryable_status_fails_fast():
    with server() as mock:
        mock.fail_next(400)
        client = LocalHttpAdapter(mock.base_url, retries=3, retry_delay_s=0.0)
        with pytest.raises(AdapterError, match="HTTP 400"):
            client.infer(make_request())
        assert mock.requests_served == 1


def test_timeout_is_reported():
    with server() as mock:
        mock.delay_next(0.8)
        client = LocalHttpAdapter(mock.base_url, timeout_s=0.05, retries=0)
        with pytest.raises(AdapterError, match="after 1 attempts"):
            client.infer(make_request())


def test_constructor_validation():
    with pytest.raises(AdapterError, match="loopback"):
        LocalHttpAdapter("http://example.com")
    with pytest.raises(AdapterError, match="loopback"):
        LocalHttpAdapter("https://127.0.0.1:8000")
    with pytest.raises(AdapterError, match="retries"):
        LocalHttpAdapter("http://127.0.0.1:9", retries=11)
    with pytest.raises(AdapterError, match="timeout"):
        LocalHttpAdapter("http://127.0.0.1:9", timeout_s=0)
    with pytest.raises(AdapterError, match="InferRequest"):
        LocalHttpAdapter("http://127.0.0.1:9").infer("nope")
