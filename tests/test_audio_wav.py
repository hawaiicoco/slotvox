"""WavAudio container validation contracts."""

import numpy as np
import pytest

from slotvox.audio.wav import WavAudio
from slotvox.errors import AudioError


def make(n=8, rate=16000):
    return WavAudio(np.zeros(n, dtype=np.float32), rate)


def test_valid_container_exposes_geometry():
    audio = make(1600)
    assert audio.n_samples == 1600
    assert audio.duration_s == pytest.approx(0.1)


def test_equality_is_elementwise_and_type_safe():
    assert make() == make()
    assert make() != WavAudio(np.ones(8, dtype=np.float32), 16000)
    assert make() != WavAudio(np.zeros(8, dtype=np.float32), 8000)
    assert make() != "not audio"


@pytest.mark.parametrize(
    ("samples", "rate"),
    [
        (np.zeros((2, 2), dtype=np.float32), 16000),
        (np.zeros(4, dtype=np.float64), 16000),
        (np.zeros(0, dtype=np.float32), 16000),
        (np.array([np.nan, 0.0], dtype=np.float32), 16000),
        (np.zeros(4, dtype=np.float32), 999),
        (np.zeros(4, dtype=np.float32), 192001),
        (np.zeros(4, dtype=np.float32), 16000.0),
        (np.zeros(4, dtype=np.float32), True),
    ],
)
def test_invalid_containers_rejected(samples, rate):
    with pytest.raises(AudioError):
        WavAudio(samples, rate)


def test_sample_rate_boundaries_accepted():
    assert make(rate=1000).sample_rate == 1000
    assert make(rate=192000).sample_rate == 192000
