"""Signal framing and window functions.

Framing never emits partial trailing frames: :func:`frame_count` and
:func:`frame_signal` only cover complete windows, which keeps downstream
feature shapes exact. :func:`frame_signal` returns a read-only strided
view, so no copy is made and callers cannot corrupt the source signal.
"""

from __future__ import annotations

import numpy as np

from slotvox.errors import ValidationError


def _check_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{name} must be a positive integer, got {value!r}")


def frame_count(n_samples: int, frame_length: int, hop: int) -> int:
    """Number of complete frames in ``n_samples`` (partial tail dropped)."""
    _check_positive_int("frame_length", frame_length)
    _check_positive_int("hop", hop)
    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise ValidationError(f"n_samples must be an int, got {n_samples!r}")
    if n_samples < 0:
        raise ValidationError(f"n_samples must be >= 0, got {n_samples}")
    if n_samples < frame_length:
        return 0
    return (n_samples - frame_length) // hop + 1


def frame_signal(samples: np.ndarray, frame_length: int, hop: int) -> np.ndarray:
    """Frame a 1-D signal into a read-only ``[n_frames, frame_length]`` view."""
    array = np.ascontiguousarray(np.asarray(samples))
    if array.ndim != 1:
        raise ValidationError(f"frame_signal expects a 1-D signal, got ndim {array.ndim}")
    count = frame_count(int(array.shape[0]), frame_length, hop)
    if count == 0:
        return np.zeros((0, frame_length), dtype=array.dtype)
    item = array.strides[0]
    return np.lib.stride_tricks.as_strided(
        array,
        shape=(count, frame_length),
        strides=(item * hop, item),
        writeable=False,
    )
