"""Framing contracts: counts, boundaries, values, view semantics."""

import numpy as np
import pytest

from slotvox.audio.framing import frame_count, frame_signal
from slotvox.errors import ValidationError


def test_frame_count_boundaries():
    assert frame_count(0, 4, 2) == 0
    assert frame_count(3, 4, 2) == 0
    assert frame_count(4, 4, 2) == 1
    assert frame_count(5, 4, 2) == 1
    assert frame_count(6, 4, 2) == 2
    assert frame_count(10, 4, 3) == 3


def test_frame_signal_values_and_shape():
    frames = frame_signal(np.arange(8, dtype=np.float32), 4, 2)
    assert frames.shape == (3, 4)
    assert np.array_equal(frames[0], [0, 1, 2, 3])
    assert np.array_equal(frames[1], [2, 3, 4, 5])
    assert np.array_equal(frames[-1], [4, 5, 6, 7])


def test_frames_view_is_read_only():
    frames = frame_signal(np.arange(8, dtype=np.float32), 4, 4)
    with pytest.raises(ValueError):
        frames[0, 0] = 9.0


def test_empty_result_when_signal_shorter_than_frame():
    frames = frame_signal(np.arange(3, dtype=np.float32), 4, 2)
    assert frames.shape == (0, 4)


def test_invalid_geometry_rejected():
    with pytest.raises(ValidationError):
        frame_count(10, 0, 2)
    with pytest.raises(ValidationError):
        frame_count(10, 4, 0)
    with pytest.raises(ValidationError):
        frame_count(-1, 4, 2)
    with pytest.raises(ValidationError):
        frame_signal(np.arange(4, dtype=np.float32), 4, 0)
    with pytest.raises(ValidationError):
        frame_signal(np.zeros((2, 2), dtype=np.float32), 2, 1)
