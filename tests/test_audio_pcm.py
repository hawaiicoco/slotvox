"""PCM16 conversion contracts: both directions, clipping, rejection."""

import numpy as np
import pytest

from slotvox.audio.pcm import float_to_pcm16, pcm16_to_float
from slotvox.errors import AudioError


def test_round_trip_within_quantization():
    samples = np.linspace(-1.0, 1.0, 101, dtype=np.float32)
    recovered = pcm16_to_float(float_to_pcm16(samples))
    assert recovered.dtype == np.float32
    assert float(np.max(np.abs(recovered - samples))) < 1e-4


def test_extremes_map_to_full_scale_and_back():
    data = float_to_pcm16(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    assert data == b"\x01\x80" + b"\x00\x00" + b"\xff\x7f"
    assert np.array_equal(pcm16_to_float(data), np.array([-1.0, 0.0, 1.0], dtype=np.float32))


def test_out_of_range_values_clipped():
    recovered = pcm16_to_float(float_to_pcm16(np.array([-2.0, 3.0], dtype=np.float32)))
    assert recovered[0] == -1.0
    assert recovered[1] == 1.0


def test_non_finite_samples_rejected():
    with pytest.raises(AudioError, match="finite"):
        float_to_pcm16(np.array([0.0, np.nan], dtype=np.float32))


def test_shape_and_dtype_guards():
    with pytest.raises(AudioError, match="ndim"):
        float_to_pcm16(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(AudioError, match="float"):
        float_to_pcm16(np.zeros(4, dtype=np.int16))
    with pytest.raises(AudioError, match="even"):
        pcm16_to_float(b"\x01")


def test_empty_signal_round_trips():
    assert float_to_pcm16(np.zeros(0, dtype=np.float32)) == b""
    assert pcm16_to_float(b"").shape == (0,)
