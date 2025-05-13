"""Mel filterbank contracts: shape, bounds, ordering, rejection."""

import numpy as np
import pytest

from slotvox.errors import ValidationError
from slotvox.features.mel import mel_filterbank


def test_shape_bounds_and_support():
    banks = mel_filterbank(16000, 512, 64, 20.0, 7600.0)
    assert banks.shape == (64, 257)
    assert banks.min() >= 0.0
    assert banks.max() <= 1.0
    assert np.all(banks.sum(axis=1) > 0.0)


def test_filters_are_ordered_by_frequency():
    banks = mel_filterbank(16000, 512, 16, 0.0, 8000.0)
    peaks = np.argmax(banks, axis=1)
    assert np.all(np.diff(peaks) >= 0)


def test_triangle_edges_are_zero_at_bounds():
    banks = mel_filterbank(8000, 16, 2, 0.0, 4000.0)
    assert banks[0, 0] == 0.0
    assert banks[-1, -1] == 0.0


def test_invalid_params_rejected():
    with pytest.raises(ValidationError):
        mel_filterbank(16000, 512, 0, 0.0, 8000.0)
    with pytest.raises(ValidationError):
        mel_filterbank(16000, 511, 8, 0.0, 8000.0)
    with pytest.raises(ValidationError):
        mel_filterbank(16000, 512, 8, 100.0, 50.0)
    with pytest.raises(ValidationError):
        mel_filterbank(16000, 512, 8, 0.0, 8001.0)
    with pytest.raises(ValidationError):
        mel_filterbank(16000.0, 512, 8, 0.0, 8000.0)


def test_filters_too_dense_for_fft_rejected():
    with pytest.raises(ValidationError, match="captures no FFT bins"):
        mel_filterbank(8000, 4, 8, 0.0, 4000.0)
