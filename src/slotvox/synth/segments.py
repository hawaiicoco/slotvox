"""Typed signal segments: the labeled atoms of synthetic utterances.

Each segment kind maps to a deterministic signal builder, so the label and
the acoustic content of a rendered segment are linked BY CONSTRUCTION.
Segments are synthetic signaling, not real speech.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from slotvox.errors import ValidationError
from slotvox.synth.signals import MAX_FORMANTS, chirp, formant_pattern, tone
from slotvox.util.jsoncanon import canonical_dumps

SIGNAL_KINDS = ("tone", "chirp", "formant")


def _finite_positive(value: Any, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValidationError(f"{name} must be a finite positive number, got {value!r}")


def _require_keys(params: dict, required: tuple[str, ...]) -> None:
    missing = [key for key in required if key not in params]
    extra = sorted(set(params) - set(required))
    if missing:
        raise ValidationError(f"segment params missing keys: {missing}")
    if extra:
        raise ValidationError(f"segment params got unknown keys: {extra}")


def validate_params(kind: str, params: Any) -> None:
    """Validate the parameter dict for a segment ``kind``, strictly.

    Structural and numeric contracts are checked here; sample-rate-relative
    bounds (Nyquist) are enforced by the builders at render time. Params
    must be canonical-JSON-native so they can be embedded in manifests.
    """
    if kind not in SIGNAL_KINDS:
        raise ValidationError(f"unknown segment kind {kind!r}; expected one of {SIGNAL_KINDS}")
    if not isinstance(params, dict):
        raise ValidationError(f"segment params must be a dict, got {type(params).__name__}")
    canonical_dumps(params)  # rejects non-JSON-native values
    if kind == "tone":
        _require_keys(params, ("freq",))
        _finite_positive(params["freq"], "freq")
    elif kind == "chirp":
        _require_keys(params, ("f_start", "f_end"))
        _finite_positive(params["f_start"], "f_start")
        _finite_positive(params["f_end"], "f_end")
        if not params["f_start"] < params["f_end"]:
            raise ValidationError(
                f"chirp requires f_start < f_end, got {params['f_start']} >= {params['f_end']}"
            )
    else:
        _require_keys(params, ("formants",))
        formants = params["formants"]
        if isinstance(formants, (str, bytes)) or not isinstance(formants, (list, tuple)):
            raise ValidationError(f"formants must be a list of frequencies, got {formants!r}")
        if not 1 <= len(formants) <= MAX_FORMANTS:
            raise ValidationError(f"formants needs 1..{MAX_FORMANTS} entries, got {len(formants)}")
        previous = 0.0
        for value in formants:
            _finite_positive(value, "formants entry")
            if float(value) <= previous:
                raise ValidationError(f"formants must be strictly ascending, got {list(formants)}")
            previous = float(value)


@dataclass(frozen=True)
class Segment:
    """A labeled signal segment with exact sample boundaries."""

    label: str
    kind: str
    start_sample: int
    end_sample: int
    params: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValidationError("Segment.label must be a non-empty string")
        validate_params(self.kind, self.params)
        for name in ("start_sample", "end_sample"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValidationError(f"Segment.{name} must be an int, got {value!r}")
        if self.start_sample < 0:
            raise ValidationError(f"Segment.start_sample must be >= 0, got {self.start_sample}")
        if self.end_sample <= self.start_sample:
            raise ValidationError(
                f"Segment.end_sample must be > start_sample, got "
                f"{self.end_sample} <= {self.start_sample}"
            )

    @property
    def n_samples(self) -> int:
        """Segment length in samples."""
        return self.end_sample - self.start_sample

    def to_dict(self) -> dict[str, Any]:
        """Manifest-ready dict (params must already be JSON-native)."""
        return {
            "label": self.label,
            "kind": self.kind,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: Any) -> Segment:
        """Reconstruct a Segment, rejecting unknown or missing keys."""
        if not isinstance(data, dict):
            raise ValidationError(f"Segment payload must be a dict, got {type(data).__name__}")
        required = {"label", "kind", "start_sample", "end_sample", "params"}
        unknown = sorted(set(data) - required)
        missing = sorted(required - set(data))
        if unknown:
            raise ValidationError(f"Segment dict got unknown keys: {unknown}")
        if missing:
            raise ValidationError(f"Segment dict is missing keys: {missing}")
        return cls(
            label=data["label"],
            kind=data["kind"],
            start_sample=data["start_sample"],
            end_sample=data["end_sample"],
            params=data["params"],
        )


def render_segment(
    kind: str,
    params: dict[str, Any],
    n_samples: int,
    sample_rate: int,
    *,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Render one segment kind into float32 samples of exactly ``n_samples``."""
    validate_params(kind, params)
    if kind == "tone":
        return tone(params["freq"], n_samples, sample_rate, amplitude=amplitude)
    if kind == "chirp":
        return chirp(
            params["f_start"], params["f_end"], n_samples, sample_rate, amplitude=amplitude
        )
    return formant_pattern(params["formants"], n_samples, sample_rate, amplitude=amplitude)
