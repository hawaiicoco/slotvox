"""Joint intent-slot training configuration (torch extra, CPU)."""

from __future__ import annotations

from dataclasses import dataclass
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

ENCODERS = ("tcn", "gru")


@dataclass(frozen=True)
@register_config
class TrainConfig(Config):
    """Training settings for the joint intent-slot model.

    ``intent_loss_weight`` and ``slot_loss_weight`` scale the two terms of
    the joint loss; at least one must be positive. Training is deterministic
    given ``seed``.
    """

    kind: ClassVar[str] = "train"

    encoder: str = "tcn"
    hidden_size: int = 64
    layers: int = 2
    dropout: float = 0.1
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 3e-3
    intent_loss_weight: float = 1.0
    slot_loss_weight: float = 1.0
    grad_clip_norm: float = 5.0
    seed: int = 20260926

    def validate(self) -> None:
        require_str("TrainConfig.encoder", self.encoder, allowed=ENCODERS)
        require_int("TrainConfig.hidden_size", self.hidden_size, minimum=4, maximum=1024)
        require_int("TrainConfig.layers", self.layers, minimum=1, maximum=4)
        require_float("TrainConfig.dropout", self.dropout, minimum=0.0, maximum=0.9)
        require_int("TrainConfig.epochs", self.epochs, minimum=1, maximum=100000)
        require_int("TrainConfig.batch_size", self.batch_size, minimum=1, maximum=4096)
        require_float("TrainConfig.learning_rate", self.learning_rate, minimum=1e-6, maximum=1.0)
        require_float(
            "TrainConfig.intent_loss_weight", self.intent_loss_weight, minimum=0.0, maximum=100.0
        )
        require_float(
            "TrainConfig.slot_loss_weight", self.slot_loss_weight, minimum=0.0, maximum=100.0
        )
        if self.intent_loss_weight + self.slot_loss_weight <= 0.0:
            raise ConfigError("TrainConfig requires a positive total loss weight")
        require_float(
            "TrainConfig.grad_clip_norm", self.grad_clip_norm, minimum=1e-6, maximum=100.0
        )
        require_int("TrainConfig.seed", self.seed, minimum=0, maximum=SEED_MAX)
