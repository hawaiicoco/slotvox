"""Partial intent hypotheses: stability, latency, revisions, parity."""

import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.model]

from slotvox.config import FeatureConfig, GenerationConfig, StreamConfig, TrainConfig  # noqa: E402
from slotvox.data.dataset import generate_dataset  # noqa: E402
from slotvox.features.frontend import log_mel  # noqa: E402
from slotvox.models.joint import JointIntentSlotModel, ModelSpec  # noqa: E402
from slotvox.schema.builtins import weather_domain  # noqa: E402
from slotvox.streaming.session import StreamSession  # noqa: E402

FEATURES = FeatureConfig(n_mels=12)


def make_model(encoder="tcn"):
    torch.manual_seed(4242)
    config = TrainConfig(encoder=encoder, hidden_size=16, layers=2)
    model = JointIntentSlotModel(ModelSpec.from_domain(weather_domain(), FEATURES, config))
    model.eval()
    return model


MODEL = make_model()
SAMPLES = (
    generate_dataset(
        GenerationConfig(
            domain="weather", seed=101, counts={"train": 6}, n_speakers=2, noise_snr_db=(40.0, 15.0)
        )
    )
    .examples[2]
    .utterance.samples
)


def session(threshold=0.7, model=MODEL):
    return StreamSession(model, FEATURES, StreamConfig(stability_threshold=threshold))


def push_all(target, samples=SAMPLES, chunk=640):
    for start in range(0, len(samples), chunk):
        target.push_audio(samples[start : start + chunk])


@pytest.mark.parametrize("encoder", ["tcn", "gru"])
def test_final_intent_matches_offline_greedy(encoder):
    model = make_model(encoder)
    with torch.no_grad():
        mel = torch.from_numpy(log_mel(SAMPLES, FEATURES)).unsqueeze(0)
        intent_logits, _ = model(mel)
        posterior = torch.softmax(intent_logits[0], dim=-1)
        offline_intent = model.spec.intents[int(posterior.argmax())]
    current = session(model=model)
    push_all(current)
    result = current.finalize()
    assert result.intent == offline_intent
    assert result.posterior == pytest.approx(float(posterior.max()), abs=1e-5)


def test_hypothesis_lifecycle():
    current = session(threshold=0.35)
    assert current.hypothesis is None
    current.push_audio(SAMPLES[:400])
    assert current.hypothesis is None  # nothing committed yet
    current.push_audio(SAMPLES[400:5760])
    hypothesis = current.hypothesis
    assert hypothesis is not None
    assert hypothesis.frames_seen == 2
    push_all(current, SAMPLES[5760:])
    # lookahead holds the last frames back until finalize()
    assert current.hypothesis.frames_seen == 48
    assert current.finalize().n_frames == 80


def test_stability_threshold_controls_flags():
    loose = session(threshold=0.35)  # measured posterior ~0.40 on this fixture
    push_all(loose)
    result = loose.finalize()
    assert result.stable
    assert result.first_stable_latency_ms == 360.0
    strict = session(threshold=0.99)
    push_all(strict)
    result = strict.finalize()
    assert not result.stable
    assert result.first_stable_latency_ms is None


def test_latency_accounting_is_audio_time():
    current = session()
    push_all(current)
    result = current.finalize()
    assert result.first_frame_latency_ms == 360.0
    assert result.audio_ms == pytest.approx(820.0)
    assert result.first_frame_latency_ms <= result.audio_ms


def test_revision_counted_when_stable_intent_changes():
    # mechanism test: drives the accumulator directly with crafted states
    current = session(threshold=1e-9)
    head = MODEL.intent_head
    hidden = MODEL.spec.train_config.hidden_size
    with torch.no_grad():
        directions = []
        for index in range(hidden):
            state = torch.zeros(1, 1, hidden)
            state[0, 0, index] = 50.0
            directions.append(int(head(state)[0].argmax()))
        other = next(i for i in range(hidden) if directions[i] != directions[0])
        first_state = torch.zeros(1, 1, hidden)
        first_state[0, 0, 0] = 50.0
        second_state = torch.zeros(1, 1, hidden)
        second_state[0, 0, other] = 50.0
    current._accumulate_intent(first_state)
    assert current.hypothesis.stable
    assert current.hypothesis.intent == MODEL.spec.intents[directions[0]]
    current._accumulate_intent(second_state)
    result = current.finalize()
    assert result.revisions == 1
    assert result.intent == MODEL.spec.intents[directions[other]]


def test_result_to_dict_golden_keys():
    current = session()
    push_all(current)
    assert sorted(current.finalize().to_dict()) == [
        "audio_ms",
        "first_frame_latency_ms",
        "first_stable_latency_ms",
        "frame_tags",
        "intent",
        "posterior",
        "revisions",
        "stable",
    ]
