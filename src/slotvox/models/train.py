"""Deterministic CPU training for the joint intent-slot model.

Seeding is explicit: every epoch's shuffle derives from
``derive_seed(config.seed, "epoch", epoch)``, torch is pinned to a single
thread, and dropout is the only in-model randomness. Two runs with the
same seed, data, and configuration produce identical histories — a
property the tests assert on a tiny fixture.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from slotvox.config import TrainConfig
from slotvox.errors import ValidationError
from slotvox.models.dataset import iterate_batches
from slotvox.models.joint import JointIntentSlotModel, joint_loss
from slotvox.models.vocab import PAD_INDEX
from slotvox.util.seed import derive_seed, make_rng

EVAL_BATCH_SIZE = 8


@dataclass(frozen=True)
class EpochRecord:
    """One epoch of honest metrics (dev fields stay None without dev data)."""

    epoch: int
    train_loss: float
    dev_loss: float | None = None
    dev_intent_accuracy: float | None = None
    dev_slot_frame_accuracy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-native dict form."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> EpochRecord:
        """Reconstruct a record, rejecting unknown or missing keys."""
        if not isinstance(data, dict):
            raise ValidationError(f"epoch record must be a dict, got {type(data).__name__}")
        required = {
            "epoch",
            "train_loss",
            "dev_loss",
            "dev_intent_accuracy",
            "dev_slot_frame_accuracy",
        }
        unknown = sorted(set(data) - required)
        missing = sorted(required - set(data))
        if unknown:
            raise ValidationError(f"epoch record got unknown keys: {unknown}")
        if missing:
            raise ValidationError(f"epoch record is missing keys: {missing}")
        return cls(**data)


def seed_everything(seed: int) -> None:
    """Pin determinism: validate the seed, seed torch, single-thread it."""
    import torch

    make_rng(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def evaluate(model, examples, *, batch_size: int = EVAL_BATCH_SIZE) -> tuple[float, float, float]:
    """Return (loss, intent accuracy, slot frame accuracy) over ``examples``."""
    import torch

    if not isinstance(model, JointIntentSlotModel):
        raise ValidationError(f"model must be a JointIntentSlotModel, got {type(model).__name__}")
    items = list(examples)
    if not items:
        raise ValidationError("evaluate needs at least one example")
    config = model.spec.train_config
    model.eval()
    total = 0.0
    count = 0
    intent_hits = 0
    slot_hits = 0
    slot_frames = 0
    with torch.no_grad():
        for batch in iterate_batches(items, batch_size):
            intent_logits, slot_logits = model(batch.mel, batch.mask)
            loss = joint_loss(
                intent_logits,
                slot_logits,
                batch.intent,
                batch.tags,
                intent_weight=config.intent_loss_weight,
                slot_weight=config.slot_loss_weight,
            )
            size = batch.size
            total += float(loss.total) * size
            count += size
            intent_hits += int((intent_logits.argmax(dim=-1) == batch.intent).sum())
            valid = batch.tags != PAD_INDEX
            slot_hits += int((slot_logits.argmax(dim=-1)[valid] == batch.tags[valid]).sum())
            slot_frames += int(valid.sum())
    return total / count, intent_hits / count, slot_hits / max(slot_frames, 1)


def train_model(
    model,
    train_examples,
    dev_examples=(),
    config: TrainConfig | None = None,
    *,
    optimizer=None,
    start_epoch: int = 0,
) -> list[EpochRecord]:
    """Train ``model`` in place; return one :class:`EpochRecord` per epoch."""
    import torch
    from torch import nn

    if not isinstance(model, JointIntentSlotModel):
        raise ValidationError(f"model must be a JointIntentSlotModel, got {type(model).__name__}")
    cfg = config if config is not None else model.spec.train_config
    if not isinstance(cfg, TrainConfig):
        raise ValidationError(f"config must be a TrainConfig, got {type(cfg).__name__}")
    train_list = list(train_examples)
    if not train_list:
        raise ValidationError("train_model needs at least one training example")
    if isinstance(start_epoch, bool) or not isinstance(start_epoch, int) or start_epoch < 0:
        raise ValidationError(f"start_epoch must be a non-negative int, got {start_epoch!r}")
    if start_epoch >= cfg.epochs:
        return []
    seed_everything(cfg.seed)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    dev_list = list(dev_examples)
    records: list[EpochRecord] = []
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        rng = make_rng(derive_seed("epoch", cfg.seed, epoch))
        running = 0.0
        seen = 0
        for batch in iterate_batches(train_list, cfg.batch_size, rng=rng):
            optimizer.zero_grad(set_to_none=True)
            intent_logits, slot_logits = model(batch.mel, batch.mask)
            loss = joint_loss(
                intent_logits,
                slot_logits,
                batch.intent,
                batch.tags,
                intent_weight=cfg.intent_loss_weight,
                slot_weight=cfg.slot_loss_weight,
            )
            loss.total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            size = batch.size
            running += float(loss.total.detach()) * size
            seen += size
        record = EpochRecord(epoch=epoch, train_loss=running / max(seen, 1))
        if dev_list:
            dev_loss, intent_accuracy, slot_accuracy = evaluate(model, dev_list)
            record = EpochRecord(
                epoch=epoch,
                train_loss=record.train_loss,
                dev_loss=dev_loss,
                dev_intent_accuracy=intent_accuracy,
                dev_slot_frame_accuracy=slot_accuracy,
            )
        records.append(record)
    return records
