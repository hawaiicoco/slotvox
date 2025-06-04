"""SNR-controlled mixing and edge fades for synthetic signals."""

from __future__ import annotations

import math

import numpy as np

from slotvox.errors import ValidationError
from slotvox.synth.signals import SIGNAL_DTYPE


def _as_1d(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValidationError(f"{name} must be 1-D, got ndim {array.ndim}")
    if array.size and not bool(np.all(np.isfinite(array))):
        raise ValidationError(f"{name} must be finite")
    return array


def apply_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix ``noise`` into ``signal`` at the target SNR (dB, signal vs noise).

    Powers are measured as mean squares. A zero-power noise leaves the
    signal unchanged (infinite SNR); a zero-power signal is rejected because
    SNR would be undefined. The sum is clipped to [-1, 1], so leave
    headroom when choosing amplitudes.
    """
    base = _as_1d(signal, "signal")
    overlay = _as_1d(noise, "noise")
    if (
        isinstance(snr_db, bool)
        or not isinstance(snr_db, (int, float))
        or not math.isfinite(snr_db)
    ):
        raise ValidationError(f"snr_db must be a finite number, got {snr_db!r}")
    if base.shape != overlay.shape:
        raise ValidationError(
            f"signal and noise must have equal length, got {base.shape[0]} vs {overlay.shape[0]}"
        )
    signal_power = float(np.mean(base**2))
    noise_power = float(np.mean(overlay**2))
    if signal_power <= 0.0:
        raise ValidationError("apply_snr requires a non-silent signal")
    if noise_power <= 0.0:
        return base.astype(SIGNAL_DTYPE)
    scale = math.sqrt(signal_power / (noise_power * 10.0 ** (float(snr_db) / 10.0)))
    return np.clip(base + scale * overlay, -1.0, 1.0).astype(SIGNAL_DTYPE)


def fade_edges(
    signal: np.ndarray,
    sample_rate: int,
    *,
    attack_ms: int = 5,
    release_ms: int = 5,
) -> np.ndarray:
    """Raised-cosine attack/release fades that remove concatenation clicks.

    The first sample of the attack ramp and the last sample of the release
    ramp are exactly zero; the middle of the signal is untouched.
    """
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValidationError(f"sample_rate must be a positive integer, got {sample_rate!r}")
    for name, value in (("attack_ms", attack_ms), ("release_ms", release_ms)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValidationError(f"{name} must be a non-negative integer, got {value!r}")
    base = _as_1d(signal, "signal")
    attack = sample_rate * attack_ms // 1000
    release = sample_rate * release_ms // 1000
    if base.size < attack + release:
        raise ValidationError(
            f"signal of {base.size} samples is too short for fades of {attack}+{release}"
        )
    out = base.copy()
    if attack:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(attack) / attack)
        out[:attack] *= ramp
    if release:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(release) / release)
        out[-release:] *= ramp[::-1]
    return out.astype(SIGNAL_DTYPE)
