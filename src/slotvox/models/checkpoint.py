"""Versioned checkpoints: strict save/load and training resume.

Checkpoint payload (schema ``slotvox.checkpoint``, version 1): the model
spec, the state dict, the epoch reached, optional optimizer state, the
epoch history, and optional metrics. Loading is strict in both
directions: unknown/missing keys, wrong schema or version, a spec that
does not match the weights, or a caller-provided expected spec mismatch
are all hard errors. ``torch.load`` runs with ``weights_only=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slotvox.errors import SchemaError, ValidationError
from slotvox.models.joint import JointIntentSlotModel, ModelSpec
from slotvox.models.train import EpochRecord

CHECKPOINT_SCHEMA_ID = "slotvox.checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1

_REQUIRED_KEYS = {
    "schema",
    "schema_version",
    "spec",
    "epoch",
    "state_dict",
    "optimizer_state",
    "history",
    "metrics",
}


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Everything restored from a checkpoint file."""

    model: JointIntentSlotModel
    spec: ModelSpec
    epoch: int
    optimizer_state: dict | None
    history: tuple[EpochRecord, ...]
    metrics: dict[str, Any] | None


def save_checkpoint(
    path: str | Path,
    model: JointIntentSlotModel,
    *,
    epoch: int,
    optimizer_state: dict | None = None,
    history=(),
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Atomically save a versioned checkpoint for ``model``."""
    import torch

    if not isinstance(model, JointIntentSlotModel):
        raise ValidationError(f"model must be a JointIntentSlotModel, got {type(model).__name__}")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValidationError(f"epoch must be a non-negative int, got {epoch!r}")
    records = list(history)
    for record in records:
        if not isinstance(record, EpochRecord):
            raise ValidationError("history must contain EpochRecord instances")
    if metrics is not None and not isinstance(metrics, dict):
        raise ValidationError(f"metrics must be a dict or None, got {type(metrics).__name__}")
    payload = {
        "schema": CHECKPOINT_SCHEMA_ID,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "spec": model.spec.to_dict(),
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer_state,
        "history": [record.to_dict() for record in records],
        "metrics": metrics,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)
    return target


def load_checkpoint(
    path: str | Path, *, expected_spec: ModelSpec | None = None
) -> LoadedCheckpoint:
    """Load and strictly validate a checkpoint; rebuilds the model."""
    import torch

    source = Path(path)
    if not source.is_file():
        raise SchemaError(f"checkpoint not found: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SchemaError(f"checkpoint unreadable: {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SchemaError("checkpoint payload must be a dict")
    unknown = sorted(set(payload) - _REQUIRED_KEYS)
    missing = sorted(_REQUIRED_KEYS - set(payload))
    if unknown:
        raise SchemaError(f"checkpoint got unknown keys: {unknown}")
    if missing:
        raise SchemaError(f"checkpoint is missing keys: {missing}")
    if payload["schema"] != CHECKPOINT_SCHEMA_ID:
        raise SchemaError(f"unknown checkpoint schema {payload['schema']!r}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise SchemaError(f"unsupported checkpoint version {payload['schema_version']!r}")
    epoch = payload["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise SchemaError(f"checkpoint epoch invalid: {epoch!r}")
    if not isinstance(payload["history"], list):
        raise SchemaError("checkpoint history must be a list")
    spec = ModelSpec.from_dict(payload["spec"])
    if expected_spec is not None and expected_spec != spec:
        raise SchemaError("checkpoint spec does not match the expected model spec")
    model = JointIntentSlotModel(spec)
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except (RuntimeError, TypeError, KeyError) as exc:
        raise SchemaError(f"checkpoint weights do not fit the spec: {exc}") from exc
    model.eval()
    return LoadedCheckpoint(
        model=model,
        spec=spec,
        epoch=epoch,
        optimizer_state=payload["optimizer_state"],
        history=tuple(EpochRecord.from_dict(item) for item in payload["history"]),
        metrics=payload["metrics"],
    )
