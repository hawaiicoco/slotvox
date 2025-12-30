"""Inference-service protocol: versioned request/response contracts.

The protocol is the boundary between slotvox and any serving stack:
requests carry bounded raw audio (JSON-native, no binary), responses
carry the greedy joint decision. Both sides are validated strictly in
BOTH directions — a service that accepts garbage, or a client that
trusts garbage, breaks the contract by design. Adapters validate
protocol behavior, not model quality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from slotvox.errors import ValidationError

REQUEST_SCHEMA_ID = "slotvox.infer-request"
REQUEST_SCHEMA_VERSION = 1
MAX_REQUEST_SAMPLES = 480_000  # 30 s at 16 kHz — a documented hard bound
SUPPORTED_SAMPLE_RATES = (8000, 16000, 22050, 44100, 48000)


def _check_request_id(value: Any) -> str:
    if not isinstance(value, str) or not value or any(ch.isspace() for ch in value):
        raise ValidationError("request_id must be a non-empty string without whitespace")
    return value


def _exact_keys(data: Any, required: set[str], label: str) -> None:
    if not isinstance(data, dict):
        raise ValidationError(f"{label} payload must be a dict, got {type(data).__name__}")
    unknown = sorted(set(data) - required)
    missing = sorted(required - set(data))
    if unknown:
        raise ValidationError(f"{label} got unknown keys: {unknown}")
    if missing:
        raise ValidationError(f"{label} is missing keys: {missing}")


@dataclass(frozen=True)
class InferRequest:
    """One bounded audio utterance to understand."""

    request_id: str
    samples: tuple[float, ...]
    sample_rate: int

    def __post_init__(self) -> None:
        _check_request_id(self.request_id)
        samples = self.samples
        if isinstance(samples, list):
            samples = tuple(samples)
            object.__setattr__(self, "samples", samples)
        if isinstance(samples, (str, bytes)) or not isinstance(samples, tuple):
            raise ValidationError("samples must be a tuple/list of numbers")
        if not samples:
            raise ValidationError("samples must be non-empty")
        if len(samples) > MAX_REQUEST_SAMPLES:
            raise ValidationError(
                f"samples exceed the bound of {MAX_REQUEST_SAMPLES} entries, got {len(samples)}"
            )
        for value in samples:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValidationError(f"samples must contain numbers, got {value!r}")
            if not math.isfinite(value):
                raise ValidationError(f"samples must be finite, got {value!r}")
        object.__setattr__(self, "samples", tuple(float(value) for value in samples))
        if (
            isinstance(self.sample_rate, bool)
            or not isinstance(self.sample_rate, int)
            or self.sample_rate not in SUPPORTED_SAMPLE_RATES
        ):
            raise ValidationError(
                f"sample_rate must be one of {SUPPORTED_SAMPLE_RATES}, got {self.sample_rate!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-native dict under the request envelope."""
        return {
            "schema": REQUEST_SCHEMA_ID,
            "schema_version": REQUEST_SCHEMA_VERSION,
            "request_id": self.request_id,
            "samples": list(self.samples),
            "sample_rate": self.sample_rate,
        }

    @classmethod
    def from_dict(cls, data: Any) -> InferRequest:
        """Reconstruct from :meth:`to_dict` output (strict)."""
        _exact_keys(
            data,
            {"schema", "schema_version", "request_id", "samples", "sample_rate"},
            "infer request",
        )
        if data["schema"] != REQUEST_SCHEMA_ID:
            raise ValidationError(f"unknown request schema {data['schema']!r}")
        if data["schema_version"] != REQUEST_SCHEMA_VERSION:
            raise ValidationError(f"unsupported request schema version {data['schema_version']!r}")
        if not isinstance(data["samples"], list):
            raise ValidationError("request 'samples' must be a list")
        return cls(
            request_id=data["request_id"],
            samples=tuple(data["samples"]),
            sample_rate=data["sample_rate"],
        )


RESPONSE_SCHEMA_ID = "slotvox.infer-response"
RESPONSE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class InferResponse:
    """The greedy joint decision for one request.

    ``frame_tags`` must be a structurally valid BIO sequence (the same
    algebra the tagging package enforces), ``posterior`` is the intent
    confidence in [0, 1], and ``model_hash`` pins which model produced
    the decision.
    """

    request_id: str
    intent: str
    posterior: float
    frame_tags: tuple[str, ...]
    model_hash: str

    def __post_init__(self) -> None:
        from slotvox.instructions.schema import check_provenance_hash
        from slotvox.tagging.bio import validate_sequence

        _check_request_id(self.request_id)
        if (
            not isinstance(self.intent, str)
            or not self.intent
            or any(ch.isspace() for ch in self.intent)
        ):
            raise ValidationError("intent must be a non-empty string without whitespace")
        if isinstance(self.posterior, bool) or not isinstance(self.posterior, (int, float)):
            raise ValidationError(f"posterior must be a number, got {self.posterior!r}")
        posterior = float(self.posterior)
        if not 0.0 <= posterior <= 1.0:
            raise ValidationError(f"posterior must be within [0, 1], got {self.posterior!r}")
        object.__setattr__(self, "posterior", posterior)
        frame_tags = self.frame_tags
        if isinstance(frame_tags, list):
            frame_tags = tuple(frame_tags)
            object.__setattr__(self, "frame_tags", frame_tags)
        if isinstance(frame_tags, (str, bytes)) or not isinstance(frame_tags, tuple):
            raise ValidationError("frame_tags must be a tuple/list of BIO tags")
        if not frame_tags:
            raise ValidationError("frame_tags must be non-empty")
        validate_sequence(frame_tags)  # strict BIO structure, both directions
        check_provenance_hash(self.model_hash, "model_hash")

    def to_dict(self) -> dict[str, Any]:
        """JSON-native dict under the response envelope."""
        return {
            "schema": RESPONSE_SCHEMA_ID,
            "schema_version": RESPONSE_SCHEMA_VERSION,
            "request_id": self.request_id,
            "intent": self.intent,
            "posterior": self.posterior,
            "frame_tags": list(self.frame_tags),
            "model_hash": self.model_hash,
        }

    @classmethod
    def from_dict(cls, data: Any) -> InferResponse:
        """Reconstruct from :meth:`to_dict` output (strict)."""
        _exact_keys(
            data,
            {
                "schema",
                "schema_version",
                "request_id",
                "intent",
                "posterior",
                "frame_tags",
                "model_hash",
            },
            "infer response",
        )
        if data["schema"] != RESPONSE_SCHEMA_ID:
            raise ValidationError(f"unknown response schema {data['schema']!r}")
        if data["schema_version"] != RESPONSE_SCHEMA_VERSION:
            raise ValidationError(f"unsupported response schema version {data['schema_version']!r}")
        if not isinstance(data["frame_tags"], list):
            raise ValidationError("response 'frame_tags' must be a list")
        return cls(
            request_id=data["request_id"],
            intent=data["intent"],
            posterior=data["posterior"],
            frame_tags=tuple(data["frame_tags"]),
            model_hash=data["model_hash"],
        )


def request_schema() -> dict[str, Any]:
    """JSON-schema subset describing request payloads (validator cross-checked)."""
    return {
        "type": "object",
        "required": ["schema", "schema_version", "request_id", "samples", "sample_rate"],
        "additionalProperties": False,
        "properties": {
            "schema": {"type": "string", "enum": [REQUEST_SCHEMA_ID]},
            "schema_version": {"type": "integer", "enum": [REQUEST_SCHEMA_VERSION]},
            "request_id": {"type": "string", "minLength": 1},
            "samples": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 1,
                "maxItems": MAX_REQUEST_SAMPLES,
            },
            "sample_rate": {"type": "integer", "enum": list(SUPPORTED_SAMPLE_RATES)},
        },
    }


def response_schema() -> dict[str, Any]:
    """JSON-schema subset describing response payloads (validator cross-checked)."""
    return {
        "type": "object",
        "required": [
            "schema",
            "schema_version",
            "request_id",
            "intent",
            "posterior",
            "frame_tags",
            "model_hash",
        ],
        "additionalProperties": False,
        "properties": {
            "schema": {"type": "string", "enum": [RESPONSE_SCHEMA_ID]},
            "schema_version": {"type": "integer", "enum": [RESPONSE_SCHEMA_VERSION]},
            "request_id": {"type": "string", "minLength": 1},
            "intent": {"type": "string", "minLength": 1},
            "posterior": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "frame_tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "model_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    }
