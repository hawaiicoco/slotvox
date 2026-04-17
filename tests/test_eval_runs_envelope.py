"""Run envelope goldens, strict validation, and hashes."""

import re

import pytest

from slotvox.errors import SchemaError
from slotvox.eval.metrics import TurnRecord
from slotvox.eval.runs import (
    RUN_SCHEMA_ID,
    RUN_SCHEMA_VERSION,
    ScoredRun,
    validate_run_envelope,
)

GOLDEN_KEYS = [
    "bootstrap",
    "confusion",
    "intent",
    "metadata",
    "record_count",
    "run_id",
    "schema",
    "schema_version",
    "slices",
    "slot_partial",
    "slot_strict",
    "turn",
]


def records():
    return (
        TurnRecord("a", "a", ("B-city", "O"), ("B-city", "O"), slices=(("noise", "clean"),)),
        TurnRecord("a", "b", ("B-city", "O"), ("O", "O"), slices=(("noise", "snr-15"),)),
    )


def test_envelope_golden_and_hash():
    run = ScoredRun.evaluate("run-one", records(), metadata={"k": "v"})
    envelope = run.to_dict()
    assert sorted(envelope) == GOLDEN_KEYS
    assert envelope["schema"] == RUN_SCHEMA_ID
    assert envelope["schema_version"] == RUN_SCHEMA_VERSION
    assert sorted(envelope["slot_strict"]) == [
        "credit",
        "f1",
        "gold_count",
        "precision",
        "pred_count",
        "recall",
    ]
    assert sorted(envelope["intent"]) == ["correct", "total"]
    assert envelope["record_count"] == 2
    assert validate_run_envelope(envelope) is envelope
    assert re.fullmatch(r"[0-9a-f]{64}", run.run_hash)
    assert run.run_hash == ScoredRun.evaluate("run-one", records(), metadata={"k": "v"}).run_hash
    assert run.run_hash != ScoredRun.evaluate("run-two", records()).run_hash


def test_envelope_mutations_rejected():
    envelope = ScoredRun.evaluate("run-one", records()).to_dict()
    with pytest.raises(SchemaError, match="unknown keys"):
        validate_run_envelope({**envelope, "extra": 1})
    with pytest.raises(SchemaError, match="missing keys"):
        validate_run_envelope({key: value for key, value in envelope.items() if key != "turn"})
    with pytest.raises(SchemaError, match="schema"):
        validate_run_envelope({**envelope, "schema": "other"})
    with pytest.raises(SchemaError, match="version"):
        validate_run_envelope({**envelope, "schema_version": 9})
    with pytest.raises(SchemaError, match="exceeds"):
        validate_run_envelope({**envelope, "intent": {"correct": 5, "total": 3}})
    with pytest.raises(SchemaError, match="dict"):
        validate_run_envelope("nope")
    with pytest.raises(SchemaError, match="record_count"):
        validate_run_envelope({**envelope, "record_count": 0})
