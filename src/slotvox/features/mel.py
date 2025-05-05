"""Mel scale conversions.

The mel scale used here is the HTK variant ``mel = 2595 * log10(1 + hz/700)``,
with :func:`mel_to_hz` as its exact inverse. Conversions are elementwise for
arrays and reject negative or non-finite inputs in both directions.
"""

from __future__ import annotations

import numpy as np

from slotvox.errors import ValidationError

MEL_REF_HZ = 700.0
MEL_SCALE = 2595.0


def hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    """Convert Hz to mel (HTK formula); elementwise for arrays."""
    array = np.asarray(hz, dtype=np.float64)
    if np.any(array < 0) or not np.all(np.isfinite(array)):
        raise ValidationError("hz_to_mel requires finite, non-negative frequencies")
    result = MEL_SCALE * np.log10(1.0 + array / MEL_REF_HZ)
    return float(result) if result.ndim == 0 else result


def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    """Convert mel to Hz (exact inverse of :func:`hz_to_mel`)."""
    array = np.asarray(mel, dtype=np.float64)
    if np.any(array < 0) or not np.all(np.isfinite(array)):
        raise ValidationError("mel_to_hz requires finite, non-negative mel values")
    result = MEL_REF_HZ * (10.0 ** (array / MEL_SCALE) - 1.0)
    return float(result) if result.ndim == 0 else result
