"""Checkpoint save/load/resume contracts, strict in both directions."""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.model]

from slotvox.config import FeatureConfig, TrainConfig  # noqa: E402
from slotvox.errors import SchemaError, ValidationError  # noqa: E402
from slotvox.models.checkpoint import (  # noqa: E402
    CHECKPOINT_SCHEMA_ID,
    load_checkpoint,
    save_checkpoint,
)
from slotvox.models.joint import JointIntentSlotModel, ModelSpec  # noqa: E402
from slotvox.models.train import EpochRecord  # noqa: E402
from slotvox.schema.builtins import weather_domain  # noqa: E402

SPEC = ModelSpec.from_domain(
    weather_domain(),
    FeatureConfig(n_mels=8),
    TrainConfig(encoder="tcn", hidden_size=8, layers=1, epochs=4),
)
RECORD = EpochRecord(
    epoch=0,
    train_loss=1.5,
    dev_loss=1.4,
    dev_intent_accuracy=0.25,
    dev_slot_frame_accuracy=0.7,
)


def fresh_model():
    torch.manual_seed(77)
    return JointIntentSlotModel(SPEC)


def test_round_trip_restores_everything(tmp_path):
    model = fresh_model()
    path = save_checkpoint(
        tmp_path / "ckpt.pt",
        model,
        epoch=2,
        optimizer_state={"lr": 0.01},
        history=(RECORD,),
        metrics={"note": "demo"},
    )
    bundle = load_checkpoint(path)
    assert bundle.spec == SPEC
    assert bundle.epoch == 2
    assert bundle.history == (RECORD,)
    assert bundle.metrics == {"note": "demo"}
    for key, value in model.state_dict().items():
        assert torch.equal(bundle.model.state_dict()[key], value)
    assert not bundle.model.training


def test_expected_spec_guard(tmp_path):
    model = fresh_model()
    path = save_checkpoint(tmp_path / "c.pt", model, epoch=0)
    other = replace(SPEC, train_config=replace(SPEC.train_config, hidden_size=16))
    with pytest.raises(SchemaError, match="does not match"):
        load_checkpoint(path, expected_spec=other)
    assert load_checkpoint(path, expected_spec=SPEC).spec == SPEC


def test_missing_and_corrupt_files_rejected(tmp_path):
    with pytest.raises(SchemaError, match="not found"):
        load_checkpoint(tmp_path / "absent.pt")
    broken = tmp_path / "broken.pt"
    broken.write_bytes(b"not a torch checkpoint at all")
    with pytest.raises(SchemaError, match="unreadable"):
        load_checkpoint(broken)


def test_tampered_weights_rejected(tmp_path):
    model = fresh_model()
    path = save_checkpoint(tmp_path / "c.pt", model, epoch=0)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    key = next(iter(payload["state_dict"]))
    payload["state_dict"][key] = torch.zeros(1)
    torch.save(payload, path)
    with pytest.raises(SchemaError, match="do not fit"):
        load_checkpoint(path)


def test_save_validation(tmp_path):
    model = fresh_model()
    with pytest.raises(ValidationError):
        save_checkpoint(tmp_path / "c.pt", model, epoch=-1)
    with pytest.raises(ValidationError):
        save_checkpoint(tmp_path / "c.pt", "not-a-model", epoch=0)
    with pytest.raises(ValidationError):
        save_checkpoint(tmp_path / "c.pt", model, epoch=0, history=("x",))
    with pytest.raises(ValidationError):
        save_checkpoint(tmp_path / "c.pt", model, epoch=0, metrics="x")


def test_epoch_record_serde_strict():
    assert EpochRecord.from_dict(RECORD.to_dict()) == RECORD
    with pytest.raises(ValidationError, match="unknown keys"):
        EpochRecord.from_dict({**RECORD.to_dict(), "x": 1})
    with pytest.raises(ValidationError, match="missing keys"):
        EpochRecord.from_dict({"epoch": 0})


def test_checkpoint_schema_constant():
    assert CHECKPOINT_SCHEMA_ID == "slotvox.checkpoint"
