"""Window function contracts with small goldens."""

import numpy as np
import pytest

from slotvox.audio.framing import WINDOW_NAMES, window
from slotvox.errors import ValidationError


def test_hann_golden_periodic():
    assert np.allclose(window("hann", 4), [0.0, 0.5, 1.0, 0.5])


def test_hamming_and_blackman_endpoints():
    assert window("hamming", 8)[0] == pytest.approx(0.08)
    assert window("blackman", 8)[0] == pytest.approx(0.0, abs=1e-12)


def test_periodic_symmetry_and_bounds():
    for name in WINDOW_NAMES:
        vec = window(name, 64)
        assert vec.shape == (64,)
        assert np.allclose(vec[1:], vec[:0:-1])
        assert vec.min() >= -1e-12
        assert vec.max() <= 1.0 + 1e-12


def test_unknown_window_rejected():
    with pytest.raises(ValidationError, match="unknown window"):
        window("kaiser", 16)


def test_invalid_length_rejected():
    with pytest.raises(ValidationError):
        window("hann", 0)
    with pytest.raises(ValidationError):
        window("hann", 4.0)
