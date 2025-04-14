"""Synthetic dialogue dataset generation configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from slotvox.config.base import (
    Config,
    register_config,
    require_float,
    require_int,
    require_str,
)
from slotvox.errors import ConfigError
from slotvox.util.seed import SEED_MAX

SPLITS = ("train", "dev", "test")
MAX_SPEAKERS = 64


@dataclass(frozen=True)
@register_config
class GenerationConfig(Config):
    """Settings for the synthetic dialogue factory.

    ``counts`` maps split names to utterance counts; at least one utterance
    must be requested overall. ``noise_snr_db`` lists signal-to-noise ratios
    (dB) that generated utterances cycle through deterministically.
    """

    kind: ClassVar[str] = "generation"

    domain: str = "weather"
    seed: int = 20260926
    counts: dict[str, int] = field(default_factory=lambda: {"train": 64, "dev": 16, "test": 16})
    n_speakers: int = 4
    noise_snr_db: tuple[float, ...] = (30.0, 15.0)

    def __post_init__(self) -> None:
        raw = self.noise_snr_db
        if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
            raise ConfigError(
                f"GenerationConfig.noise_snr_db must be a list or tuple, got {type(raw).__name__}"
            )
        object.__setattr__(self, "noise_snr_db", tuple(raw))
        super().__post_init__()

    def validate(self) -> None:
        require_str("GenerationConfig.domain", self.domain)
        if not self.domain:
            raise ConfigError("GenerationConfig.domain must be non-empty")
        require_int("GenerationConfig.seed", self.seed, minimum=0, maximum=SEED_MAX)
        if not isinstance(self.counts, dict):
            raise ConfigError(
                f"GenerationConfig.counts must be a dict, got {type(self.counts).__name__}"
            )
        unknown = set(self.counts) - set(SPLITS)
        if unknown:
            raise ConfigError(f"GenerationConfig.counts got unknown splits: {sorted(unknown)}")
        for split, count in self.counts.items():
            require_int(f"GenerationConfig.counts[{split!r}]", count, minimum=0)
        if sum(self.counts.values()) < 1:
            raise ConfigError("GenerationConfig.counts must request at least one utterance")
        require_int("GenerationConfig.n_speakers", self.n_speakers, minimum=1, maximum=MAX_SPEAKERS)
        if not self.noise_snr_db:
            raise ConfigError("GenerationConfig.noise_snr_db must be non-empty")
        for index, snr in enumerate(self.noise_snr_db):
            require_float(
                f"GenerationConfig.noise_snr_db[{index}]", snr, minimum=0.0, maximum=100.0
            )
