"""Log-mel front-end configuration.

``WINDOWS`` is re-exported from the audio layer so configuration and
implementation can never disagree about the supported window names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from slotvox.audio.framing import WINDOW_NAMES as WINDOWS
from slotvox.config.base import (
    Config,
    register_config,
    require_float,
    require_int,
    require_str,
)
from slotvox.errors import ConfigError


@dataclass(frozen=True)
@register_config
class FeatureConfig(Config):
    """Log-mel front-end settings.

    Frames are ``frame_ms`` long, advance by ``hop_ms``, and are zero-padded
    to ``n_fft`` before the FFT. ``fmax`` must stay within the Nyquist limit
    (``sample_rate / 2``).
    """

    kind: ClassVar[str] = "feature"

    sample_rate: int = 16000
    frame_ms: int = 25
    hop_ms: int = 10
    n_fft: int = 512
    n_mels: int = 64
    fmin: float = 20.0
    fmax: float = 7600.0
    window: str = "hann"

    def validate(self) -> None:
        require_int("FeatureConfig.sample_rate", self.sample_rate, minimum=1000, maximum=192000)
        require_int("FeatureConfig.frame_ms", self.frame_ms, minimum=1)
        require_int("FeatureConfig.hop_ms", self.hop_ms, minimum=1)
        if self.hop_ms > self.frame_ms:
            raise ConfigError(
                f"FeatureConfig.hop_ms must be <= frame_ms, got {self.hop_ms} > {self.frame_ms}"
            )
        require_int("FeatureConfig.n_fft", self.n_fft, minimum=2)
        if self.n_fft & (self.n_fft - 1):
            raise ConfigError(f"FeatureConfig.n_fft must be a power of two, got {self.n_fft}")
        require_int("FeatureConfig.n_mels", self.n_mels, minimum=1, maximum=512)
        nyquist = self.sample_rate / 2
        require_float("FeatureConfig.fmin", self.fmin, minimum=0.0)
        require_float("FeatureConfig.fmax", self.fmax, minimum=0.0)
        if self.fmin >= self.fmax:
            raise ConfigError(f"FeatureConfig.fmin must be < fmax, got {self.fmin} >= {self.fmax}")
        if self.fmax > nyquist:
            raise ConfigError(f"FeatureConfig.fmax must be <= Nyquist ({nyquist}), got {self.fmax}")
        require_str("FeatureConfig.window", self.window, allowed=WINDOWS)

    @property
    def frame_length(self) -> int:
        """Frame length in samples."""
        return self.sample_rate * self.frame_ms // 1000

    @property
    def hop_length(self) -> int:
        """Hop size in samples."""
        return self.sample_rate * self.hop_ms // 1000
