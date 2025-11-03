"""Joint intent-slot models over log-mel frames (torch extra).

Vocabulary building (:mod:`slotvox.models.vocab`) is torch-free. Every
other name is exported lazily through module ``__getattr__``: importing
``slotvox.models`` works without the extra, and touching a torch-backed
name raises a helpful ImportError naming the extra. All models here train
on slotvox's synthetic datasets only — no pretrained weights, no
downloads, CPU-only by design.
"""

from __future__ import annotations

import importlib
from typing import Any

from slotvox.models.vocab import PAD_INDEX, Vocab, build_intent_vocab, build_tag_vocab

_TORCH_MODULES = {
    "CHECKPOINT_SCHEMA_ID": "slotvox.models.checkpoint",
    "EncodedExample": "slotvox.models.dataset",
    "EpochRecord": "slotvox.models.train",
    "FrameBatch": "slotvox.models.dataset",
    "GRUEncoder": "slotvox.models.encoder",
    "IntentHead": "slotvox.models.heads",
    "JointIntentSlotModel": "slotvox.models.joint",
    "JointLoss": "slotvox.models.joint",
    "LoadedCheckpoint": "slotvox.models.checkpoint",
    "ModelSpec": "slotvox.models.joint",
    "SlotHead": "slotvox.models.heads",
    "TCNEncoder": "slotvox.models.encoder",
    "build_encoder": "slotvox.models.encoder",
    "collate": "slotvox.models.dataset",
    "encode_split": "slotvox.models.dataset",
    "evaluate": "slotvox.models.train",
    "iterate_batches": "slotvox.models.dataset",
    "joint_loss": "slotvox.models.joint",
    "load_checkpoint": "slotvox.models.checkpoint",
    "save_checkpoint": "slotvox.models.checkpoint",
    "seed_everything": "slotvox.models.train",
    "train_model": "slotvox.models.train",
}

__all__ = [
    "CHECKPOINT_SCHEMA_ID",
    "EncodedExample",
    "EpochRecord",
    "FrameBatch",
    "GRUEncoder",
    "IntentHead",
    "JointIntentSlotModel",
    "JointLoss",
    "LoadedCheckpoint",
    "ModelSpec",
    "PAD_INDEX",
    "SlotHead",
    "TCNEncoder",
    "Vocab",
    "build_encoder",
    "build_intent_vocab",
    "build_tag_vocab",
    "collate",
    "encode_split",
    "evaluate",
    "iterate_batches",
    "joint_loss",
    "load_checkpoint",
    "save_checkpoint",
    "seed_everything",
    "train_model",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve torch-backed exports with a helpful error when absent."""
    module_name = _TORCH_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'slotvox.models' has no attribute {name!r}")
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"slotvox.models.{name} requires the torch extra: "
            "pip install 'slotvox[torch]' "
            "(CPU wheels: --index-url https://download.pytorch.org/whl/cpu)"
        ) from exc
    return getattr(importlib.import_module(module_name), name)
