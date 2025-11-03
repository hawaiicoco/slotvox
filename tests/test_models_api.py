"""Golden pin for the models package API surface (lazy torch exports)."""

import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.model]

import slotvox.models as models  # noqa: E402
from slotvox.models.joint import JointIntentSlotModel  # noqa: E402

GOLDEN_ALL = [
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


def test_models_api_surface_is_pinned():
    assert sorted(models.__all__) == GOLDEN_ALL
    for name in GOLDEN_ALL:
        assert hasattr(models, name)


def test_lazy_names_resolve_to_real_objects():
    assert models.JointIntentSlotModel is JointIntentSlotModel
    assert models.PAD_INDEX == -100
    assert models.CHECKPOINT_SCHEMA_ID == "slotvox.checkpoint"


def test_unknown_attribute_rejected():
    with pytest.raises(AttributeError):
        models.nonexistent_name
