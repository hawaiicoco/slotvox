"""Chunked streaming-inference configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from slotvox.config.base import Config, register_config, require_float, require_int
from slotvox.errors import ConfigError


@dataclass(frozen=True)
@register_config
class StreamConfig(Config):
    """Streaming settings.

    ``buffer_capacity_ms`` is a hard bound on retained audio: the session
    consumes and discards, never growing beyond it. ``lookahead_ms`` of audio
    is held before slot decisions commit, trading latency for stability.
    ``stability_threshold`` is the posterior an intent hypothesis must reach
    to be flagged stable.
    """

    kind: ClassVar[str] = "stream"

    sample_rate: int = 16000
    chunk_ms: int = 160
    buffer_capacity_ms: int = 3200
    lookahead_ms: int = 320
    stability_threshold: float = 0.7

    def validate(self) -> None:
        require_int("StreamConfig.sample_rate", self.sample_rate, minimum=1000, maximum=192000)
        require_int("StreamConfig.chunk_ms", self.chunk_ms, minimum=10, maximum=10000)
        require_int(
            "StreamConfig.buffer_capacity_ms",
            self.buffer_capacity_ms,
            minimum=4 * self.chunk_ms,
        )
        require_int("StreamConfig.lookahead_ms", self.lookahead_ms, minimum=0)
        if self.lookahead_ms > self.buffer_capacity_ms:
            raise ConfigError(
                "StreamConfig.lookahead_ms must be <= buffer_capacity_ms, "
                f"got {self.lookahead_ms} > {self.buffer_capacity_ms}"
            )
        require_float("StreamConfig.stability_threshold", self.stability_threshold)
        if not 0.0 < self.stability_threshold <= 1.0:
            raise ConfigError(
                "StreamConfig.stability_threshold must be within (0, 1], "
                f"got {self.stability_threshold}"
            )

    @property
    def chunk_samples(self) -> int:
        """Chunk size in samples."""
        return self.sample_rate * self.chunk_ms // 1000
