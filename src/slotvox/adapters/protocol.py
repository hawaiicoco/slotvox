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
