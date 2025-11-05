"""Bounded audio buffer for chunked streaming.

The buffer never grows beyond ``capacity`` samples: appending beyond it
is an explicit error, not silent truncation, so callers must consume.
Consumed audio is dropped immediately — a long stream through a small
buffer keeps memory flat (a property the tests enforce).
"""

from __future__ import annotations

import numpy as np

from slotvox.errors import StreamingError


class AudioBuffer:
    """A FIFO sample buffer with a hard capacity bound (float32 mono)."""

    def __init__(self, capacity: int):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise StreamingError(f"capacity must be a positive int, got {capacity!r}")
        self._capacity = capacity
        self._chunks: list[np.ndarray] = []
        self._pending = 0

    @property
    def capacity(self) -> int:
        """Hard bound in samples."""
        return self._capacity

    @property
    def pending(self) -> int:
        """Unconsumed sample count (always <= capacity)."""
        return self._pending

    def append(self, samples: np.ndarray) -> None:
        """Add float mono samples; rejects anything that would overflow."""
        array = np.asarray(samples)
        if array.ndim != 1:
            raise StreamingError(f"buffer expects 1-D samples, got ndim {array.ndim}")
        if array.dtype not in (np.float32, np.float64):
            raise StreamingError(f"buffer expects float samples, got {array.dtype}")
        if array.size and not bool(np.all(np.isfinite(array))):
            raise StreamingError("buffer samples must be finite")
        array = array.astype(np.float32, copy=False)
        if array.size > self._capacity:
            raise StreamingError(
                f"append of {array.size} samples exceeds capacity {self._capacity}"
            )
        if self._pending + array.size > self._capacity:
            raise StreamingError(
                f"buffer overflow: {self._pending} + {array.size} > {self._capacity}; consume first"
            )
        if array.size:
            self._chunks.append(array)
            self._pending += int(array.size)

    def peek(self) -> np.ndarray:
        """Copy of all pending samples, oldest first."""
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks)

    def consume(self, count: int) -> np.ndarray:
        """Remove and return the oldest ``count`` samples."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise StreamingError(f"count must be a non-negative int, got {count!r}")
        if count > self._pending:
            raise StreamingError(f"cannot consume {count} samples, only {self._pending} pending")
        out = self.peek()[:count]
        remaining = count
        while remaining > 0 and self._chunks:
            head = self._chunks[0]
            if head.size <= remaining:
                remaining -= int(head.size)
                self._chunks.pop(0)
            else:
                self._chunks[0] = head[remaining:]
                remaining = 0
        self._pending -= count
        return out

    def clear(self) -> None:
        """Drop all pending samples."""
        self._chunks.clear()
        self._pending = 0
