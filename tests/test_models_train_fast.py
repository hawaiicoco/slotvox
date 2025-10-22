"""Fast training contracts: improvement, determinism, resume numbering.

Tiny fixture (4 train / 2 dev synthetic examples) — runs in seconds and is
NOT the honest end-to-end improvement test (that one is marked slow).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.model]

from slotvox.config import FeatureConfig, GenerationConfig, TrainConfig  # noqa: E402
from slotvox.data.alignment import frame_tags, frame_token_map  # noqa: E402
from slotvox.data.dataset import generate_dataset  # noqa: E402
from slotvox.errors import ValidationError  # noqa: E402
from slotvox.features.frontend import log_mel  # noqa: E402
from slotvox.models.dataset import EncodedExample  # noqa: E402
from slotvox.models.joint import JointIntentSlotModel, ModelSpec  # noqa: E402
from slotvox.models.train import evaluate, seed_everything, train_model  # noqa: E402
from slotvox.models.vocab import Vocab  # noqa: E402
from slotvox.schema.builtins import weather_domain  # noqa: E402

FEATURES = FeatureConfig(n_mels=8)
TRAIN_CFG = TrainConfig(
    encoder="tcn",
    hidden_size=16,
    layers=1,
    epochs=12,
    batch_size=4,
    learning_rate=0.01,
    seed=5,
)
SPEC = ModelSpec.from_domain(weather_domain(), FEATURES, TRAIN_CFG)
INTENTS = Vocab(SPEC.intents)
TAGS = Vocab(SPEC.tags)


def encoded(split):
    dataset = generate_dataset(
        GenerationConfig(
            domain="weather",
            seed=55,
            counts={"train": 4, "dev": 2},
            n_speakers=2,
            noise_snr_db=(40.0,),
        )
    )
    out = []
    for example in dataset.split_examples(split):
        mel = log_mel(example.utterance.samples, FEATURES)
        token_map = frame_token_map(example.utterance, FEATURES)
        names = frame_tags(example.annotation.tags, token_map)
        out.append(
            EncodedExample(
                utterance_id=example.utterance_id,
                mel=mel,
                frame_tag_indices=np.asarray(TAGS.indices(names), dtype=np.int64),
                intent_index=INTENTS.index(example.annotation.intent),
                frame_tag_names=names,
            )
        )
    return out


TRAIN = encoded("train")
DEV = encoded("dev")


def fresh_model():
    torch.manual_seed(123)
    return JointIntentSlotModel(SPEC)


def test_loss_decreases_and_dev_metrics_are_reported():
    records = train_model(fresh_model(), TRAIN, DEV)
    assert [record.epoch for record in records] == list(range(12))
    assert records[-1].train_loss < records[0].train_loss
    assert all(0.0 <= record.dev_intent_accuracy <= 1.0 for record in records)
    assert all(0.0 <= record.dev_slot_frame_accuracy <= 1.0 for record in records)
    assert all(record.dev_loss is not None for record in records)


def test_training_is_deterministic():
    first = train_model(fresh_model(), TRAIN, DEV)
    second = train_model(fresh_model(), TRAIN, DEV)
    assert first == second


def test_start_epoch_controls_numbering():
    records = train_model(fresh_model(), TRAIN, DEV, start_epoch=9)
    assert [record.epoch for record in records] == [9, 10, 11]
    assert train_model(fresh_model(), TRAIN, DEV, start_epoch=12) == []


def test_evaluate_bounds_and_validation():
    loss, intent_acc, slot_acc = evaluate(fresh_model(), DEV)
    assert loss > 0.0
    assert 0.0 <= intent_acc <= 1.0
    assert 0.0 <= slot_acc <= 1.0
    with pytest.raises(ValidationError):
        evaluate(fresh_model(), [])
    with pytest.raises(ValidationError):
        evaluate("not-a-model", DEV)


def test_train_model_validation():
    with pytest.raises(ValidationError):
        train_model(fresh_model(), [])
    with pytest.raises(ValidationError):
        train_model("not-a-model", TRAIN)
    with pytest.raises(ValidationError):
        train_model(fresh_model(), TRAIN, DEV, start_epoch=-1)


def test_seed_everything_validates_and_pins_threads():
    with pytest.raises(ValidationError):
        seed_everything(-1)
    seed_everything(3)
    assert torch.get_num_threads() == 1
