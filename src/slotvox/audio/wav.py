"""Validated in-memory audio container."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slotvox.errors import AudioError

MIN_SAMPLE_RATE = 1000
MAX_SAMPLE_RATE = 192000


@dataclass(frozen=True, eq=False)
class WavAudio:
    """Mono float32 audio with a validated sample rate.

    Instances are intentionally strict: only 1-D float32 finite arrays with
    an int sample rate inside [MIN_SAMPLE_RATE, MAX_SAMPLE_RATE] are
    accepted. Equality compares sample rate and samples elementwise.
    """

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples)
        if samples.ndim != 1:
            raise AudioError(f"WavAudio samples must be 1-D, got ndim {samples.ndim}")
        if samples.dtype != np.float32:
            raise AudioError(f"WavAudio samples must be float32, got {samples.dtype}")
        if samples.size == 0:
            raise AudioError("WavAudio samples must be non-empty")
        if not bool(np.all(np.isfinite(samples))):
            raise AudioError("WavAudio samples must be finite")
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise AudioError(f"sample_rate must be an int, got {self.sample_rate!r}")
        if not MIN_SAMPLE_RATE <= self.sample_rate <= MAX_SAMPLE_RATE:
            raise AudioError(
                f"sample_rate must be within [{MIN_SAMPLE_RATE}, {MAX_SAMPLE_RATE}], "
                f"got {self.sample_rate}"
            )
        object.__setattr__(self, "samples", samples)

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return int(self.samples.shape[0])

    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        return self.n_samples / self.sample_rate

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WavAudio):
            return NotImplemented
        return self.sample_rate == other.sample_rate and bool(
            np.array_equal(self.samples, other.samples)
        )
