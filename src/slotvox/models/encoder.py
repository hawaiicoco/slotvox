"""Compact causal speech encoders over log-mel frames (torch extra, CPU).

``tcn`` stacks dilated CAUSAL convolutions: frame ``t`` sees only frames
``<= t``, and the receptive field is finite and computed exactly — the
property bounded-buffer streaming relies on for offline equivalence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as tf
from torch import nn

from slotvox.errors import ValidationError

KERNEL_SIZE = 3


def _check_dims(input_size: int, hidden_size: int, layers: int, dropout: float) -> None:
    for name, value in (
        ("input_size", input_size),
        ("hidden_size", hidden_size),
        ("layers", layers),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValidationError(f"{name} must be a positive int, got {value!r}")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise ValidationError(f"dropout must be a number, got {dropout!r}")
    if not 0.0 <= float(dropout) < 1.0:
        raise ValidationError(f"dropout must be within [0, 1), got {dropout!r}")


class CausalConvBlock(nn.Module):
    """Dilated causal conv1d -> GELU -> dropout, left-padded only."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        if kernel_size < 2:
            raise ValidationError("kernel_size must be >= 2 for causal padding")
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.conv(tf.pad(x, (self.left_pad, 0)))))


class TCNEncoder(nn.Module):
    """``[B, T, input] -> [B, T, hidden]``; strictly causal conv stack."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        layers: int,
        dropout: float,
        kernel_size: int = KERNEL_SIZE,
    ):
        super().__init__()
        _check_dims(input_size, hidden_size, layers, dropout)
        self.input_proj = nn.Conv1d(input_size, hidden_size, 1)
        self.blocks = nn.ModuleList(
            [
                CausalConvBlock(hidden_size, kernel_size, 2**index, dropout)
                for index in range(layers)
            ]
        )
        self.receptive_field = 1 + sum((kernel_size - 1) * 2**index for index in range(layers))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 3:
            raise ValidationError(
                f"encoder expects [B, T, F] frames, got shape {tuple(frames.shape)}"
            )
        hidden = self.input_proj(frames.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        return hidden.transpose(1, 2)
