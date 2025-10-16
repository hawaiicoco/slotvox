"""Joint model, ModelSpec, and joint_loss contracts."""

import re

import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.model]

from slotvox.config import FeatureConfig, TrainConfig  # noqa: E402
from slotvox.errors import ValidationError  # noqa: E402
from slotvox.models.joint import (  # noqa: E402
    JointIntentSlotModel,
    ModelSpec,
    joint_loss,
)
from slotvox.models.vocab import PAD_INDEX  # noqa: E402
from slotvox.schema.builtins import weather_domain  # noqa: E402


def spec(**overrides):
    data = {
        "feature_config": FeatureConfig(n_mels=12),
        "train_config": TrainConfig(encoder="tcn", hidden_size=8, layers=1),
    }
    data.update(overrides)
    return ModelSpec.from_domain(weather_domain(), **data)


def test_spec_from_domain_builds_consistent_vocabs():
    model_spec = spec()
    assert model_spec.intents[0] == "query-weather"
    assert model_spec.tags[0] == "O"
    assert "B-city" in model_spec.tags
    assert len(model_spec.tags) == 9


def test_spec_round_trip_and_hash():
    model_spec = spec()
    assert ModelSpec.from_dict(model_spec.to_dict()) == model_spec
    assert re.fullmatch(r"[0-9a-f]{64}", model_spec.model_hash)
    other = spec(train_config=TrainConfig(encoder="gru", hidden_size=8, layers=1))
    assert other.model_hash != model_spec.model_hash


def test_spec_validation():
    with pytest.raises(ValidationError, match="start with"):
        ModelSpec("weather", ("a",), ("B-city",), FeatureConfig(n_mels=4), TrainConfig())
    with pytest.raises(ValidationError, match="unique"):
        ModelSpec("weather", ("a", "a"), ("O",), FeatureConfig(n_mels=4), TrainConfig())
    with pytest.raises(ValidationError):
        ModelSpec("nope", ("a",), ("O",), FeatureConfig(n_mels=4), TrainConfig())
    payload = spec().to_dict()
    with pytest.raises(ValidationError, match="unknown keys"):
        ModelSpec.from_dict({**payload, "extra": 1})
    with pytest.raises(ValidationError, match="schema"):
        ModelSpec.from_dict({**payload, "schema": "other"})


def test_forward_shapes_and_greedy_names():
    torch.manual_seed(6)
    model = JointIntentSlotModel(spec())
    mel = torch.randn(2, 20, 12)
    intent_logits, slot_logits = model(mel)
    assert intent_logits.shape == (2, 3)
    assert slot_logits.shape == (2, 20, 9)
    intent, tags = model.predict_greedy(torch.randn(20, 12))
    assert intent in spec().intents
    assert len(tags) == 20
    assert all(tag in spec().tags for tag in tags)


def test_predict_greedy_validates_features():
    model = JointIntentSlotModel(spec())
    with pytest.raises(ValidationError, match="12 bins"):
        model.predict_greedy(torch.randn(20, 8))
    with pytest.raises(ValidationError, match=r"\[T, F\]"):
        model.predict_greedy(torch.randn(2, 20, 12))


def test_loss_weighting_is_linear_in_components():
    torch.manual_seed(7)
    logits_i = torch.randn(3, 3, requires_grad=False)
    logits_s = torch.randn(3, 5, 9)
    intent_targets = torch.tensor([0, 1, 2])
    slot_targets = torch.randint(0, 9, (3, 5))
    both = joint_loss(
        logits_i, logits_s, intent_targets, slot_targets, intent_weight=1.0, slot_weight=1.0
    )
    only_i = joint_loss(
        logits_i, logits_s, intent_targets, slot_targets, intent_weight=1.0, slot_weight=0.0
    )
    only_s = joint_loss(
        logits_i, logits_s, intent_targets, slot_targets, intent_weight=0.0, slot_weight=1.0
    )
    assert torch.allclose(only_i.total, only_i.intent)
    assert torch.allclose(only_s.total, only_s.slot)
    assert torch.allclose(both.total, both.intent + both.slot)


def test_all_padded_slot_targets_stay_finite():
    logits_i = torch.randn(1, 3)
    logits_s = torch.randn(1, 4, 9)
    loss = joint_loss(
        logits_i,
        logits_s,
        torch.tensor([1]),
        torch.full((1, 4), PAD_INDEX, dtype=torch.long),
        intent_weight=1.0,
        slot_weight=1.0,
    )
    assert float(loss.slot) == 0.0
    assert torch.isfinite(loss.total)


def test_loss_input_validation():
    logits_i = torch.randn(2, 3)
    logits_s = torch.randn(2, 4, 9)
    good_i = torch.tensor([0, 1])
    good_s = torch.randint(0, 9, (2, 4))
    with pytest.raises(ValidationError, match="positive total weight"):
        joint_loss(logits_i, logits_s, good_i, good_s, intent_weight=0.0, slot_weight=0.0)
    with pytest.raises(ValidationError):
        joint_loss(logits_i, logits_s, good_i, good_s, intent_weight=-1.0, slot_weight=1.0)
    with pytest.raises(ValidationError, match=r"\[0, 3\)"):
        joint_loss(
            logits_i, logits_s, torch.tensor([0, 3]), good_s, intent_weight=1.0, slot_weight=1.0
        )
    with pytest.raises(ValidationError, match=r"\[0, 9\)"):
        joint_loss(
            logits_i, logits_s, good_i, torch.full((2, 4), 99), intent_weight=1.0, slot_weight=1.0
        )
    with pytest.raises(ValidationError, match="shape"):
        joint_loss(
            logits_i,
            logits_s,
            good_i,
            torch.randint(0, 9, (2, 5)),
            intent_weight=1.0,
            slot_weight=1.0,
        )


def test_gradients_flow_through_joint_loss():
    torch.manual_seed(8)
    model = JointIntentSlotModel(spec())
    mel = torch.randn(2, 10, 12)
    intent_logits, slot_logits = model(mel)
    loss = joint_loss(
        intent_logits,
        slot_logits,
        torch.tensor([0, 2]),
        torch.randint(0, 9, (2, 10)),
        intent_weight=0.5,
        slot_weight=1.0,
    ).total
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
