"""Offline replay adapter: fixtures in, validated decisions out.

The replay adapter serves canned responses from a fixture store. It is
the offline test double for the inference protocol: fixture payloads go
through the SAME strict validation as live traffic in both directions,
so replay exercises protocol behavior without any model or network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slotvox.adapters.protocol import InferRequest, InferResponse
from slotvox.errors import AdapterError, ValidationError
from slotvox.schema.serialize import read_json, write_json_atomic

FIXTURE_SCHEMA_ID = "slotvox.adapter-fixtures"
FIXTURE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Fixture:
    """One request/response pair with matching ids (validated)."""

    request: InferRequest
    response: InferResponse

    def __post_init__(self) -> None:
        if not isinstance(self.request, InferRequest):
            raise AdapterError(
                f"request must be an InferRequest, got {type(self.request).__name__}"
            )
        if not isinstance(self.response, InferResponse):
            raise AdapterError(
                f"response must be an InferResponse, got {type(self.response).__name__}"
            )
        if self.request.request_id != self.response.request_id:
            raise AdapterError(
                f"fixture request/response id mismatch: {self.request.request_id!r} "
                f"vs {self.response.request_id!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-native pair form."""
        return {"request": self.request.to_dict(), "response": self.response.to_dict()}


def fixture_from_dict(data: Any) -> Fixture:
    """Rebuild one fixture (strict; protocol validation applies)."""
    if not isinstance(data, dict) or set(data) != {"request", "response"}:
        raise AdapterError("fixture must be a dict with exactly 'request' and 'response'")
    try:
        return Fixture(
            request=InferRequest.from_dict(data["request"]),
            response=InferResponse.from_dict(data["response"]),
        )
    except ValidationError as exc:
        raise AdapterError(f"fixture is invalid: {exc}") from exc


def save_fixtures(fixtures, path: str | Path) -> Path:
    """Atomically write a fixture store envelope."""
    items = _require_fixtures(fixtures)
    payload = {
        "schema": FIXTURE_SCHEMA_ID,
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "fixtures": [fixture.to_dict() for fixture in items],
    }
    return write_json_atomic(Path(path), payload)


def load_fixtures(path: str | Path) -> tuple[Fixture, ...]:
    """Read and validate a fixture store envelope (strict)."""
    payload = read_json(Path(path))
    if not isinstance(payload, dict):
        raise AdapterError("fixture store must contain a JSON object")
    if set(payload) != {"schema", "schema_version", "fixtures"}:
        raise AdapterError(f"fixture store envelope keys are wrong: {sorted(set(payload))}")
    if payload["schema"] != FIXTURE_SCHEMA_ID:
        raise AdapterError(f"unknown fixture schema {payload['schema']!r}")
    if payload["schema_version"] != FIXTURE_SCHEMA_VERSION:
        raise AdapterError(f"unsupported fixture schema version {payload['schema_version']!r}")
    if not isinstance(payload["fixtures"], list):
        raise AdapterError("fixture store 'fixtures' must be a list")
    fixtures = []
    for index, row in enumerate(payload["fixtures"]):
        try:
            fixtures.append(fixture_from_dict(row))
        except AdapterError as exc:
            raise AdapterError(f"fixture {index} is invalid: {exc}") from exc
    return tuple(fixtures)


def _require_fixtures(fixtures) -> list[Fixture]:
    if isinstance(fixtures, (str, bytes)) or not isinstance(fixtures, (list, tuple)):
        raise AdapterError(f"fixtures must be a list/tuple, got {type(fixtures).__name__}")
    for fixture in fixtures:
        if not isinstance(fixture, Fixture):
            raise AdapterError(f"fixtures must contain Fixture, got {type(fixture).__name__}")
    return list(fixtures)
