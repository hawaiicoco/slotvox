"""Synthetic utterances: labeled signal segments assembled into one clip.

NOT real speech. Every utterance is a deterministic sequence of
tone/chirp/formant segments; labels attach to segments BY CONSTRUCTION, so
annotations can never disagree with the audio. Metrics computed on this data
measure whether models learn the constructed patterns — nothing more.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from slotvox.audio.wav import WavAudio
from slotvox.errors import ValidationError
from slotvox.synth.mix import apply_snr, fade_edges
from slotvox.synth.segments import Segment, render_segment, validate_params
from slotvox.synth.signals import _check_rate, noise, silence
from slotvox.util.seed import make_rng


@dataclass(frozen=True)
class SyntheticUtterance:
    """A rendered clip plus its by-construction segment labels."""

    audio: WavAudio
    segments: tuple[Segment, ...]
    snr_db: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.audio, WavAudio):
            raise ValidationError(
                f"SyntheticUtterance.audio must be WavAudio, got {type(self.audio).__name__}"
            )
        segments = self.segments
        if isinstance(segments, list):
            segments = tuple(segments)
            object.__setattr__(self, "segments", segments)
        if not isinstance(segments, tuple):
            raise ValidationError("SyntheticUtterance.segments must be a tuple/list of Segment")
        cursor = 0
        for segment in segments:
            if not isinstance(segment, Segment):
                raise ValidationError("SyntheticUtterance.segments must contain Segment instances")
            if segment.start_sample < cursor:
                raise ValidationError("segments must be sorted by start_sample and non-overlapping")
            if segment.end_sample > self.audio.n_samples:
                raise ValidationError(
                    f"segment {segment.label!r} exceeds utterance bounds "
                    f"({segment.end_sample} > {self.audio.n_samples})"
                )
            cursor = segment.end_sample
        if self.snr_db is not None:
            if isinstance(self.snr_db, bool) or not isinstance(self.snr_db, (int, float)):
                raise ValidationError(
                    f"snr_db must be a finite number or None, got {self.snr_db!r}"
                )
            object.__setattr__(self, "snr_db", float(self.snr_db))
            if not math.isfinite(self.snr_db):
                raise ValidationError(f"snr_db must be finite, got {self.snr_db!r}")

    @property
    def samples(self) -> np.ndarray:
        """Float32 mono samples."""
        return self.audio.samples

    @property
    def sample_rate(self) -> int:
        """Sample rate in Hz."""
        return self.audio.sample_rate

    @property
    def n_samples(self) -> int:
        """Total sample count."""
        return self.audio.n_samples

    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        return self.audio.duration_s

    def segment_view(self, segment: Segment) -> np.ndarray:
        """Exact sample slice for ``segment`` (must belong to this utterance)."""
        if segment not in self.segments:
            raise ValidationError("segment does not belong to this utterance")
        return self.audio.samples[segment.start_sample : segment.end_sample]

    def to_dict(self) -> dict[str, Any]:
        """Boundary metadata (samples live in the WAV payload, not here)."""
        return {
            "sample_rate": self.sample_rate,
            "n_samples": self.n_samples,
            "snr_db": self.snr_db,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, data: Any, samples: np.ndarray) -> SyntheticUtterance:
        """Reconstruct metadata around ``samples``, strictly validating keys."""
        if not isinstance(data, dict):
            raise ValidationError(f"utterance payload must be a dict, got {type(data).__name__}")
        required = {"sample_rate", "n_samples", "snr_db", "segments"}
        unknown = sorted(set(data) - required)
        missing = sorted(required - set(data))
        if unknown:
            raise ValidationError(f"utterance dict got unknown keys: {unknown}")
        if missing:
            raise ValidationError(f"utterance dict is missing keys: {missing}")
        if not isinstance(data["segments"], list):
            raise ValidationError("utterance 'segments' must be a list")
        audio = WavAudio(
            samples=np.asarray(samples, dtype=np.float32), sample_rate=data["sample_rate"]
        )
        if audio.n_samples != data["n_samples"]:
            raise ValidationError(
                f"utterance n_samples {data['n_samples']} != provided samples {audio.n_samples}"
            )
        return cls(
            audio=audio,
            segments=tuple(Segment.from_dict(item) for item in data["segments"]),
            snr_db=data["snr_db"],
        )


@dataclass(frozen=True)
class SegmentPlan:
    """Pre-render recipe for one segment of a synthetic utterance."""

    label: str
    kind: str
    duration_ms: int
    params: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValidationError("SegmentPlan.label must be a non-empty string")
        validate_params(self.kind, self.params)
        if isinstance(self.duration_ms, bool) or not isinstance(self.duration_ms, int):
            raise ValidationError(
                f"SegmentPlan.duration_ms must be an int, got {self.duration_ms!r}"
            )
        if not 1 <= self.duration_ms <= 60000:
            raise ValidationError(
                f"SegmentPlan.duration_ms must be within [1, 60000], got {self.duration_ms}"
            )


def render_utterance(
    plans: list[SegmentPlan] | tuple[SegmentPlan, ...],
    sample_rate: int,
    *,
    gap_ms: int = 20,
    snr_db: float | None = None,
    seed: int = 0,
    amplitude: float = 0.8,
) -> SyntheticUtterance:
    """Render plans into one utterance with exact by-construction boundaries.

    Layout is ``[gap][seg1][gap][seg2]...[gap]``. When ``snr_db`` is set,
    white noise drawn from ``make_rng(seed)`` is mixed over the whole clip;
    boundaries stay valid because mixing preserves length.
    """
    _check_rate(sample_rate)
    if isinstance(plans, (str, bytes)) or not isinstance(plans, (list, tuple)) or not plans:
        raise ValidationError("render_utterance needs a non-empty list/tuple of SegmentPlan")
    for plan in plans:
        if not isinstance(plan, SegmentPlan):
            raise ValidationError(
                f"plans must contain SegmentPlan instances, got {type(plan).__name__}"
            )
    if isinstance(gap_ms, bool) or not isinstance(gap_ms, int) or gap_ms < 0:
        raise ValidationError(f"gap_ms must be a non-negative integer, got {gap_ms!r}")
    if snr_db is not None and (
        isinstance(snr_db, bool)
        or not isinstance(snr_db, (int, float))
        or not math.isfinite(snr_db)
    ):
        raise ValidationError(f"snr_db must be a finite number or None, got {snr_db!r}")
    rng = make_rng(seed)

    gap_n = sample_rate * gap_ms // 1000
    pieces = [silence(gap_n)]
    segments: list[Segment] = []
    cursor = gap_n
    for plan in plans:
        n = sample_rate * plan.duration_ms // 1000
        if n < 1:
            raise ValidationError(
                f"segment {plan.label!r} renders to zero samples; increase duration_ms"
            )
        rendered = render_segment(plan.kind, plan.params, n, sample_rate, amplitude=amplitude)
        pieces.append(fade_edges(rendered, sample_rate))
        pieces.append(silence(gap_n))
        segments.append(
            Segment(
                label=plan.label,
                kind=plan.kind,
                start_sample=cursor,
                end_sample=cursor + n,
                params=dict(plan.params),
            )
        )
        cursor += n + gap_n
    samples = np.concatenate(pieces)
    if snr_db is not None:
        overlay = noise(samples.shape[0], sample_rate, amplitude=0.5, rng=rng)
        samples = apply_snr(samples, overlay, float(snr_db))
    return SyntheticUtterance(
        audio=WavAudio(samples=samples, sample_rate=sample_rate),
        segments=tuple(segments),
        snr_db=None if snr_db is None else float(snr_db),
    )
