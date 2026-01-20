"""ReplayAdapter: lookup, duplicate rejection, and type enforcement."""

import pytest

from slotvox.adapters.protocol import InferRequest
from slotvox.adapters.replay import ReplayAdapter
from slotvox.errors import AdapterError
from tests.test_adapters_helpers import make_fixture, make_request


def test_infer_returns_the_fixture_response():
    adapter = ReplayAdapter((make_fixture("req-1"), make_fixture("req-2")))
    assert len(adapter) == 2
    assert adapter.request_ids == ("req-1", "req-2")
    response = adapter.infer(make_request("req-2"))
    assert response.request_id == "req-2"
    assert response.intent == "query-weather"


def test_unknown_request_id():
    adapter = ReplayAdapter((make_fixture(),))
    with pytest.raises(AdapterError, match="no fixture"):
        adapter.infer(InferRequest("req-404", (0.1,), 16000))


def test_duplicate_ids_rejected():
    with pytest.raises(AdapterError, match="duplicate"):
        ReplayAdapter((make_fixture(), make_fixture()))


def test_type_enforcement_both_ways():
    with pytest.raises(AdapterError, match="Fixture"):
        ReplayAdapter(("nope",))
    adapter = ReplayAdapter((make_fixture(),))
    with pytest.raises(AdapterError, match="InferRequest"):
        adapter.infer("nope")
