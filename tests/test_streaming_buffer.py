"""AudioBuffer contracts: FIFO behavior, hard bound, validation."""

import numpy as np
import pytest

from slotvox.errors import StreamingError
from slotvox.streaming.buffer import AudioBuffer


def test_append_consume_roundtrip():
    buffer = AudioBuffer(10)
    buffer.append(np.arange(6, dtype=np.float32))
    assert buffer.pending == 6
    assert np.array_equal(buffer.peek(), np.arange(6, dtype=np.float32))
    consumed = buffer.consume(4)
    assert np.array_equal(consumed, [0, 1, 2, 3])
    assert buffer.pending == 2
    assert np.array_equal(buffer.peek(), [4, 5])


def test_capacity_enforced_in_both_directions():
    buffer = AudioBuffer(8)
    with pytest.raises(StreamingError, match="exceeds capacity"):
        buffer.append(np.zeros(9, dtype=np.float32))
    buffer.append(np.zeros(5, dtype=np.float32))
    with pytest.raises(StreamingError, match="overflow"):
        buffer.append(np.zeros(4, dtype=np.float32))
    with pytest.raises(StreamingError, match="cannot consume"):
        buffer.consume(6)


def test_long_stream_never_exceeds_capacity():
    buffer = AudioBuffer(100)
    total = 0
    for step in range(1000):
        buffer.append(np.full(50, step % 7, dtype=np.float32))
        assert buffer.pending <= buffer.capacity
        total += int(buffer.consume(50).size)
    assert total == 50000
    assert buffer.pending == 0


def test_input_validation():
    buffer = AudioBuffer(4)
    with pytest.raises(StreamingError, match="1-D"):
        buffer.append(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(StreamingError, match="float"):
        buffer.append(np.zeros(2, dtype=np.int16))
    with pytest.raises(StreamingError, match="finite"):
        buffer.append(np.array([np.nan, 1.0], dtype=np.float32))
    for bad in (0, -3, True, 4.0):
        with pytest.raises(StreamingError):
            AudioBuffer(bad)
    with pytest.raises(StreamingError):
        buffer.consume(-1)


def test_float64_input_is_cast_to_float32():
    buffer = AudioBuffer(4)
    buffer.append(np.array([0.5, 0.25], dtype=np.float64))
    assert buffer.peek().dtype == np.float32


def test_clear_and_empty_peek():
    buffer = AudioBuffer(4)
    assert buffer.peek().shape == (0,)
    buffer.append(np.ones(3, dtype=np.float32))
    buffer.clear()
    assert buffer.pending == 0
