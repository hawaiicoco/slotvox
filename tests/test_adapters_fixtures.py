"""Fixture store: envelope strictness and roundtrips."""

import pytest

from slotvox.adapters.protocol import InferRequest, InferResponse
from slotvox.adapters.replay import Fixture, fixture_from_dict, load_fixtures, save_fixtures
from slotvox.errors import AdapterError
from slotvox.schema.serialize import read_json, write_json_atomic

HASH = "11" * 32


def make_fixture(request_id="req-1"):
    return Fixture(
        InferRequest(request_id, (0.1, 0.2), 16000),
        InferResponse(request_id, "query-weather", 0.7, ("O", "O"), HASH),
    )


def test_roundtrip(tmp_path):
    fixtures = (make_fixture("req-1"), make_fixture("req-2"))
    path = save_fixtures(fixtures, tmp_path / "fixtures.json")
    assert load_fixtures(path) == fixtures


def test_envelope_keys_and_strictness(tmp_path):
    path = save_fixtures((make_fixture(),), tmp_path / "f.json")
    payload = read_json(path)
    assert sorted(payload) == ["fixtures", "schema", "schema_version"]
    write_json_atomic(path, {**payload, "schema": "other"})
    with pytest.raises(AdapterError, match="schema"):
        load_fixtures(path)
    write_json_atomic(path, {key: value for key, value in payload.items() if key != "fixtures"})
    with pytest.raises(AdapterError, match="envelope"):
        load_fixtures(path)


def test_fixture_pairing_and_row_validation():
    with pytest.raises(AdapterError, match="mismatch"):
        Fixture(
            InferRequest("req-1", (0.1,), 16000),
            InferResponse("req-2", "a", 0.5, ("O",), HASH),
        )
    with pytest.raises(AdapterError, match="exactly"):
        fixture_from_dict({"request": {}})
    bad = make_fixture().to_dict()
    bad["response"]["intent"] = ""
    with pytest.raises(AdapterError, match="invalid"):
        fixture_from_dict(bad)


def test_store_input_validation(tmp_path):
    with pytest.raises(AdapterError, match="Fixture"):
        save_fixtures(["nope"], tmp_path / "f.json")
    with pytest.raises(AdapterError, match="list/tuple"):
        save_fixtures("nope", tmp_path / "f.json")
