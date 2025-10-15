"""Intent and slot heads over encoder frame states."""

from __future__ import annotations

import torch
from torch import nn

from slotvox.errors import ValidationError


def _positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(f"{name} must be a positive int, got {value!r}")


class IntentHead(nn.Module):
    """Mean-pooled frame states -> utterance intent logits.

    ``from_accumulated`` applies the same linear layer to a running
    sum/count, so a streaming session that accumulates frame states gets
    the same decision as offline mean pooling by construction.
    """

    def __init__(self, hidden_size: int, num_intents: int):
        super().__init__()
        _positive("hidden_size", hidden_size)
        _positive("num_intents", num_intents)
        self.dense = nn.Linear(hidden_size, num_intents)

    def forward(self, states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, T, H]`` states (+ optional ``[B, T]`` bool mask) -> ``[B, I]``."""
        if states.ndim != 3:
            raise ValidationError(f"states must be [B, T, H], got shape {tuple(states.shape)}")
        if mask is None:
            pooled = states.mean(dim=1)
        else:
            if mask.ndim != 2 or mask.shape != states.shape[:2]:
                raise ValidationError(
                    f"mask must be [B, T] matching states, got {tuple(mask.shape)}"
                )
            if mask.dtype != torch.bool:
                raise ValidationError(f"mask must be bool, got {mask.dtype}")
            weights = mask.unsqueeze(-1).to(states.dtype)
            counts = weights.sum(dim=1).clamp(min=1.0)
            pooled = (states * weights).sum(dim=1) / counts
        return self.dense(pooled)

    def from_accumulated(self, state_sum: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        """Intent logits from a running ``[B, H]`` sum and ``[B]`` count."""
        if state_sum.ndim != 2:
            raise ValidationError(f"state_sum must be [B, H], got shape {tuple(state_sum.shape)}")
        if count.ndim != 1 or count.shape[0] != state_sum.shape[0]:
            raise ValidationError("count must be [B] matching state_sum")
        pooled = state_sum / count.clamp(min=1).unsqueeze(-1).to(state_sum.dtype)
        return self.dense(pooled)


class SlotHead(nn.Module):
    """Per-frame BIO tag logits: ``[B, T, H]`` -> ``[B, T, num_tags]``."""

    def __init__(self, hidden_size: int, num_tags: int):
        super().__init__()
        _positive("hidden_size", hidden_size)
        _positive("num_tags", num_tags)
        self.dense = nn.Linear(hidden_size, num_tags)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3:
            raise ValidationError(f"states must be [B, T, H], got shape {tuple(states.shape)}")
        return self.dense(states)
