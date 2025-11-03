"""Honest end-to-end improvement on synthetic train -> dev splits.

Measured in this repository (weather zh, seed 777, 192 train / 48 dev
utterances, speaker-disjoint splits, TCN 2x32, 10 epochs, lr 3e-3,
n_mels 24, SNR cycle 40/20/10 dB):

- train loss 2.03 -> 0.12, dev loss 1.13 -> 0.21
- dev intent accuracy 0.67 -> 1.00
- dev slot frame accuracy 0.87 -> 0.96
- dev partial-credit frame-span F1 0.81 -> 0.91

These numbers describe the SYNTHETIC task only — signals whose labels are
attached by construction. They are not evidence about real speech.
"""

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.model, pytest.mark.slow]

from slotvox.config import FeatureConfig, GenerationConfig, TrainConfig  # noqa: E402
from slotvox.data.alignment import frame_span_tags, frame_token_map  # noqa: E402
from slotvox.data.dataset import generate_dataset  # noqa: E402
from slotvox.features.frontend import log_mel  # noqa: E402
from slotvox.models.dataset import EncodedExample, iterate_batches  # noqa: E402
from slotvox.models.joint import JointIntentSlotModel, ModelSpec  # noqa: E402
from slotvox.models.train import train_model  # noqa: E402
from slotvox.models.vocab import Vocab  # noqa: E402
from slotvox.schema.builtins import weather_domain  # noqa: E402
from slotvox.tagging.bio import repair_sequence, tags_to_spans  # noqa: E402

FEATURES = FeatureConfig(n_mels=24)
CFG = TrainConfig(
    encoder="tcn",
    hidden_size=32,
    layers=2,
    epochs=10,
    batch_size=16,
    learning_rate=3e-3,
    seed=777,
    dropout=0.1,
)
SPEC = ModelSpec.from_domain(weather_domain(), FEATURES, CFG)
TAGS = Vocab(SPEC.tags)
INTENTS = Vocab(SPEC.intents)


@pytest.fixture(scope="module")
def splits():
    dataset = generate_dataset(
        GenerationConfig(
            domain="weather",
            seed=777,
            counts={"train": 192, "dev": 48},
            n_speakers=4,
            noise_snr_db=(40.0, 20.0, 10.0),
        )
    )

    def encode(split):
        out = []
        for example in dataset.split_examples(split):
            mel = log_mel(example.utterance.samples, FEATURES)
            names = frame_span_tags(
                example.annotation.tags, frame_token_map(example.utterance, FEATURES)
            )
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

    return encode("train"), encode("dev")


def fresh_model(spec=SPEC):
    torch.manual_seed(31337)
    return JointIntentSlotModel(spec)


def partial_span_f1(pred_names, gold_names):
    """Partial-credit F1: same label + nonzero overlap, greedy 1-to-1."""
    pred = tags_to_spans(repair_sequence(tuple(pred_names), "promote"))
    gold = tags_to_spans(tuple(gold_names))
    if not pred and not gold:
        return 1.0
    pairs = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gold):
            if p.label == g.label:
                overlap = min(p.end, g.end) - max(p.start, g.start)
                if overlap > 0:
                    pairs.append((overlap, pi, gi))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_pred, used_gold, matched = set(), set(), 0
    for _, pi, gi in pairs:
        if pi not in used_pred and gi not in used_gold:
            used_pred.add(pi)
            used_gold.add(gi)
            matched += 1
    precision = matched / max(len(pred), 1)
    recall = matched / max(len(gold), 1)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def dev_span_f1(model, dev):
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in iterate_batches(dev, 8):
            _, slot_logits = model(batch.mel, batch.mask)
            predicted = slot_logits.argmax(dim=-1)
            for row in range(batch.size):
                valid = batch.mask[row].bool()
                names = [SPEC.tags[i] for i in predicted[row][valid].tolist()]
                gold = [SPEC.tags[i] for i in batch.tags[row][valid].tolist()]
                scores.append(partial_span_f1(names, gold))
    return float(np.mean(scores))


def test_train_to_dev_improvement_is_real(splits):
    train, dev = splits
    baseline = fresh_model()
    train_model(baseline, train, dev, replace(CFG, epochs=1))
    baseline_f1 = dev_span_f1(baseline, dev)

    model = fresh_model()
    records = train_model(model, train, dev)
    final_f1 = dev_span_f1(model, dev)

    first, last = records[0], records[-1]
    assert last.train_loss < 0.5 * first.train_loss  # measured 2.03 -> 0.12
    assert last.dev_loss < first.dev_loss  # measured 1.13 -> 0.21
    assert last.dev_intent_accuracy >= 0.9  # measured 1.00
    assert last.dev_intent_accuracy > first.dev_intent_accuracy  # 0.67 -> 1.00
    assert last.dev_slot_frame_accuracy >= 0.93  # measured 0.96
    assert last.dev_slot_frame_accuracy > first.dev_slot_frame_accuracy
    assert final_f1 >= 0.85  # measured 0.91
    assert final_f1 > baseline_f1  # measured 0.81 -> 0.91
    assert [record.epoch for record in records] == list(range(CFG.epochs))


def test_gru_encoder_also_improves(splits):
    train, dev = splits
    gru_cfg = replace(CFG, encoder="gru", epochs=6)
    model = fresh_model(replace(SPEC, train_config=gru_cfg))
    records = train_model(model, train, dev, gru_cfg)
    assert records[-1].train_loss < records[0].train_loss
    assert records[-1].dev_intent_accuracy >= 0.5
