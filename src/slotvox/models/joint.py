"""Joint intent-slot model with a configurable weighted loss.

The model glues a causal encoder to two heads and defines the training
contract: ``forward`` for logits, ``predict_greedy`` for named decisions,
and :func:`joint_loss` for the weighted objective. ``ModelSpec`` is the
versioned identity of a trained model — vocabularies, feature geometry,
and training configuration — and round-trips through canonical JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as tf
from torch import nn

from slotvox.config import FeatureConfig, TrainConfig
from slotvox.errors import ValidationError
from slotvox.models.encoder import build_encoder
from slotvox.models.heads import IntentHead, SlotHead
from slotvox.models.vocab import PAD_INDEX, build_intent_vocab, build_tag_vocab
from slotvox.schema.builtins import builtin_domain
from slotvox.schema.domains import DomainSpec
from slotvox.tagging.bio import OUTSIDE
from slotvox.util.jsoncanon import stable_hash

MODEL_SPEC_SCHEMA_ID = "slotvox.model-spec"
MODEL_SPEC_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModelSpec:
    """Versioned identity of a joint model."""

    domain_id: str
    intents: tuple[str, ...]
    tags: tuple[str, ...]
    feature_config: FeatureConfig
    train_config: TrainConfig

    def __post_init__(self) -> None:
        builtin_domain(self.domain_id)  # strict: unknown domains rejected
        for field_name in ("intents", "tags"):
            value = getattr(self, field_name)
            if isinstance(value, list):
                object.__setattr__(self, field_name, tuple(value))
            value = getattr(self, field_name)
            if not isinstance(value, tuple) or not value:
                raise ValidationError(f"ModelSpec.{field_name} must be a non-empty tuple")
            if not all(isinstance(name, str) and name for name in value):
                raise ValidationError(f"ModelSpec.{field_name} must contain non-empty strings")
            if len(set(value)) != len(value):
                raise ValidationError(f"ModelSpec.{field_name} must be unique")
        if self.tags[0] != OUTSIDE:
            raise ValidationError(
                f"ModelSpec.tags must start with {OUTSIDE!r}, got {self.tags[0]!r}"
            )
        if not isinstance(self.feature_config, FeatureConfig):
            raise ValidationError("ModelSpec.feature_config must be a FeatureConfig")
        if not isinstance(self.train_config, TrainConfig):
            raise ValidationError("ModelSpec.train_config must be a TrainConfig")

    @classmethod
    def from_domain(
        cls,
        domain: DomainSpec,
        feature_config: FeatureConfig | None = None,
        train_config: TrainConfig | None = None,
    ) -> ModelSpec:
        """Build a consistent spec from a domain (vocabs derived, not typed)."""
        if not isinstance(domain, DomainSpec):
            raise ValidationError(f"domain must be a DomainSpec, got {type(domain).__name__}")
        return cls(
            domain_id=domain.domain,
            intents=build_intent_vocab(domain),
            tags=build_tag_vocab(domain),
            feature_config=feature_config or FeatureConfig(),
            train_config=train_config or TrainConfig(),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-native dict under the model-spec envelope."""
        return {
            "schema": MODEL_SPEC_SCHEMA_ID,
            "schema_version": MODEL_SPEC_SCHEMA_VERSION,
            "domain_id": self.domain_id,
            "intents": list(self.intents),
            "tags": list(self.tags),
            "feature_config": self.feature_config.to_dict(),
            "train_config": self.train_config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Any) -> ModelSpec:
        """Reconstruct a spec, rejecting unknown keys and bad envelopes."""
        from slotvox.config import config_from_dict

        if not isinstance(data, dict):
            raise ValidationError(f"model spec payload must be a dict, got {type(data).__name__}")
        required = {
            "schema",
            "schema_version",
            "domain_id",
            "intents",
            "tags",
            "feature_config",
            "train_config",
        }
        unknown = sorted(set(data) - required)
        missing = sorted(required - set(data))
        if unknown:
            raise ValidationError(f"model spec got unknown keys: {unknown}")
        if missing:
            raise ValidationError(f"model spec is missing keys: {missing}")
        if data["schema"] != MODEL_SPEC_SCHEMA_ID:
            raise ValidationError(f"unknown model spec schema {data['schema']!r}")
        if data["schema_version"] != MODEL_SPEC_SCHEMA_VERSION:
            raise ValidationError(f"unsupported model spec version {data['schema_version']!r}")
        feature = config_from_dict(data["feature_config"])
        train = config_from_dict(data["train_config"])
        if not isinstance(feature, FeatureConfig) or not isinstance(train, TrainConfig):
            raise ValidationError("model spec configs must be FeatureConfig and TrainConfig")
        return cls(
            domain_id=data["domain_id"],
            intents=tuple(data["intents"]),
            tags=tuple(data["tags"]),
            feature_config=feature,
            train_config=train,
        )

    @property
    def model_hash(self) -> str:
        """Provenance hash of the full spec."""
        return stable_hash(self.to_dict())


class JointIntentSlotModel(nn.Module):
    """Causal encoder + intent head + frame-level slot head."""

    def __init__(self, spec: ModelSpec):
        super().__init__()
        if not isinstance(spec, ModelSpec):
            raise ValidationError(f"spec must be a ModelSpec, got {type(spec).__name__}")
        self.spec = spec
        self.encoder = build_encoder(spec.train_config, spec.feature_config.n_mels)
        self.intent_head = IntentHead(spec.train_config.hidden_size, len(spec.intents))
        self.slot_head = SlotHead(spec.train_config.hidden_size, len(spec.tags))

    def encode(self, mel: torch.Tensor, hidden=None):
        """Frame states ``[B, T, H]`` (GRU also returns its hidden state)."""
        if self.spec.train_config.encoder == "gru":
            states, hidden = self.encoder(mel, hidden)
            return states, hidden
        return self.encoder(mel), None

    def forward(self, mel: torch.Tensor, mask: torch.Tensor | None = None):
        """``[B, T, F]`` mel -> (intent logits ``[B, I]``, slot logits ``[B, T, G]``)."""
        states, _ = self.encode(mel)
        return self.intent_head(states, mask), self.slot_head(states)

    @torch.no_grad()
    def predict_greedy(self, mel: torch.Tensor) -> tuple[str, tuple[str, ...]]:
        """Greedy decode one utterance: (intent name, per-frame tag names)."""
        if mel.ndim == 2:
            mel = mel.unsqueeze(0)
        if mel.ndim != 3 or mel.shape[0] != 1:
            raise ValidationError(
                f"predict_greedy expects [T, F] or [1, T, F], got {tuple(mel.shape)}"
            )
        expected = self.spec.feature_config.n_mels
        if mel.shape[-1] != expected:
            raise ValidationError(f"mel features must have {expected} bins, got {mel.shape[-1]}")
        was_training = self.training
        self.eval()
        try:
            intent_logits, slot_logits = self.forward(mel)
        finally:
            if was_training:
                self.train()
        intent = self.spec.intents[int(intent_logits.argmax(dim=-1)[0])]
        tags = tuple(self.spec.tags[index] for index in slot_logits.argmax(dim=-1)[0].tolist())
        return intent, tags


@dataclass(frozen=True)
class JointLoss:
    """Weighted joint loss with its two components (for honest logging)."""

    total: torch.Tensor
    intent: torch.Tensor
    slot: torch.Tensor


def joint_loss(
    intent_logits: torch.Tensor,
    slot_logits: torch.Tensor,
    intent_targets: torch.Tensor,
    slot_targets: torch.Tensor,
    *,
    intent_weight: float,
    slot_weight: float,
) -> JointLoss:
    """Weighted intent CE + slot CE (padded frames ignored via PAD_INDEX).

    A batch whose slot targets are entirely padding contributes zero slot
    loss (documented boundary; avoids NaN from an all-ignored reduction).
    """
    for name, weight in (("intent_weight", intent_weight), ("slot_weight", slot_weight)):
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValidationError(f"{name} must be a number, got {weight!r}")
        if weight < 0.0:
            raise ValidationError(f"{name} must be >= 0, got {weight!r}")
    if float(intent_weight) + float(slot_weight) <= 0.0:
        raise ValidationError("joint_loss requires a positive total weight")
    if intent_logits.ndim != 2 or intent_targets.ndim != 1:
        raise ValidationError("intent logits must be [B, I] and targets [B]")
    if intent_targets.shape[0] != intent_logits.shape[0]:
        raise ValidationError("intent target batch does not match logits")
    num_intents = intent_logits.shape[1]
    if intent_targets.numel() and (
        int(intent_targets.min()) < 0 or int(intent_targets.max()) >= num_intents
    ):
        raise ValidationError(f"intent targets must be within [0, {num_intents})")
    if slot_logits.ndim != 3 or slot_targets.ndim != 2:
        raise ValidationError("slot logits must be [B, T, G] and targets [B, T]")
    if slot_targets.shape != slot_logits.shape[:2]:
        raise ValidationError("slot target shape does not match logits")
    num_tags = slot_logits.shape[2]
    if slot_targets.numel():
        low = (
            int(slot_targets[slot_targets != PAD_INDEX].min())
            if bool((slot_targets != PAD_INDEX).any())
            else 0
        )
        high = int(slot_targets.max())
        if low < 0 or high >= num_tags:
            raise ValidationError(f"slot targets must be within [0, {num_tags}) or PAD_INDEX")
    intent_component = tf.cross_entropy(intent_logits, intent_targets)
    valid = slot_targets != PAD_INDEX
    if bool(valid.any()):
        slot_component = tf.cross_entropy(slot_logits[valid], slot_targets[valid])
    else:
        slot_component = torch.zeros((), dtype=slot_logits.dtype)
    total = float(intent_weight) * intent_component + float(slot_weight) * slot_component
    return JointLoss(total=total, intent=intent_component, slot=slot_component)
