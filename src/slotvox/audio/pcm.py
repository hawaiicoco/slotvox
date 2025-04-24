"""PCM16 <-> float32 conversion for mono audio.

slotvox stores audio as float32 in [-1, 1] and exchanges it with files as
little-endian signed 16-bit PCM. Conversions are explicit in both
directions; encoding clips out-of-range values (a documented lossy step)
and rejects non-finite samples, which have no PCM representation.
"""

from __future__ import annotations

import numpy as np

from slotvox.errors import AudioError

PCM_SCALE = 32767.0


def float_to_pcm16(samples: np.ndarray) -> bytes:
    """Convert float mono samples in [-1, 1] to little-endian PCM16 bytes."""
    array = np.asarray(samples)
    if array.ndim != 1:
        raise AudioError(f"expected mono samples with ndim 1, got ndim {array.ndim}")
    if array.dtype.kind != "f":
        raise AudioError(f"expected float samples, got dtype {array.dtype}")
    if array.size and not bool(np.all(np.isfinite(array))):
        raise AudioError("samples must be finite; NaN/Inf have no PCM16 representation")
    clipped = np.clip(array.astype(np.float64), -1.0, 1.0)
    return (clipped * PCM_SCALE).astype("<i2").tobytes()


def pcm16_to_float(data: bytes) -> np.ndarray:
    """Convert little-endian PCM16 bytes to float32 samples in [-1, 1]."""
    if len(data) % 2:
        raise AudioError(f"PCM16 byte length must be even, got {len(data)}")
    integers = np.frombuffer(data, dtype="<i2")
    return (integers.astype(np.float64) / PCM_SCALE).astype(np.float32)
