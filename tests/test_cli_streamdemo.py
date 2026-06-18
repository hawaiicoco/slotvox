"""`slotvox stream-demo`: offline mechanics demo on an untrained model."""

import json

import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.model]

from slotvox.cli import main  # noqa: E402

BASE = ["stream-demo", "--seed", "3", "--hidden", "8", "--n-mels", "12"]


def test_payload_fields(capsys):
    assert main(BASE + ["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["frames"] > 0
    assert isinstance(payload["intent"], str) and payload["intent"]
    assert payload["first_frame_latency_ms"] is not None
    assert payload["revisions"] >= 0
    assert payload["audio_ms"] > 0
    assert payload["span_count"] >= 0


def test_deterministic_for_same_seed(capsys):
    assert main(BASE + ["--json"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(BASE + ["--json"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert first == second


def test_human_output_is_honest(capsys):
    assert main(BASE) == 0
    out = capsys.readouterr().out
    assert "untrained" in out.lower()
    assert "first frame at:" in out
