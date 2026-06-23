"""`slotvox serve-mock` loopback service tests (fully offline)."""

import contextlib
import io
import json
import threading
import time

from slotvox.adapters.http import LocalHttpAdapter
from slotvox.adapters.protocol import InferRequest, InferResponse
from slotvox.adapters.replay import Fixture, save_fixtures
from slotvox.cli import main

HASH = "5e" * 32


def write_fixtures(tmp_path):
    path = tmp_path / "fixtures.json"
    fixture = Fixture(
        InferRequest("req-1", (0.1, 0.2, 0.3), 16000),
        InferResponse("req-1", "query-weather", 0.9, ("O", "B-city", "O"), HASH),
    )
    save_fixtures((fixture,), path)
    return path


def wait_for_url(buffer, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = buffer.getvalue()
        if "serving at " in text:
            return text.split("serving at ", 1)[1].splitlines()[0].strip()
        time.sleep(0.05)
    raise AssertionError("server did not announce its URL in time")


def test_serve_mock_bounded_roundtrip(tmp_path):
    fixtures = write_fixtures(tmp_path)
    buffer = io.StringIO()
    result = {}

    def run():
        with contextlib.redirect_stdout(buffer):
            result["rc"] = main(
                [
                    "serve-mock",
                    "--fixtures",
                    str(fixtures),
                    "--requests",
                    "1",
                    "--timeout-s",
                    "15",
                    "--json",
                ]
            )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    url = wait_for_url(buffer)
    response = LocalHttpAdapter(url).infer(InferRequest("req-1", (0.1, 0.2, 0.3), 16000))
    thread.join(timeout=15)
    assert result["rc"] == 0
    assert response.intent == "query-weather"
    tail = buffer.getvalue().split("serving at ", 1)[1].split("\n", 1)[1]
    payload = json.loads(tail)
    assert payload["requests_served"] == 1
    assert payload["fixtures"] == 1


def test_serve_mock_timeout_exits_runtime(tmp_path, capsys):
    fixtures = write_fixtures(tmp_path)
    code = main(
        ["serve-mock", "--fixtures", str(fixtures), "--requests", "1", "--timeout-s", "0.4"]
    )
    assert code == 4
    assert "timed out" in capsys.readouterr().err


def test_serve_mock_missing_fixtures_is_data_error(tmp_path, capsys):
    code = main(["serve-mock", "--fixtures", str(tmp_path / "nope.json"), "--requests", "1"])
    assert code == 3
    assert "not found" in capsys.readouterr().err


def test_serve_mock_argument_validation(tmp_path, capsys):
    fixtures = write_fixtures(tmp_path)
    code = main(["serve-mock", "--fixtures", str(fixtures), "--requests", "-1"])
    assert code == 2
    assert "--requests" in capsys.readouterr().err
