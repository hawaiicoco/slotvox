"""Resumable stream state: strict serde and exact resume equivalence."""

import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.model]

from slotvox.config import FeatureConfig, GenerationConfig, StreamConfig, TrainConfig  # noqa: E402
from slotvox.data.dataset import generate_dataset  # noqa: E402
from slotvox.errors import StreamingError  # noqa: E402
from slotvox.models.joint import JointIntentSlotModel, ModelSpec  # noqa: E402
from slotvox.schema.builtins import weather_domain  # noqa: E402
from slotvox.streaming.session import StreamSession  # noqa: E402
from slotvox.util.jsoncanon import stable_hash  # noqa: E402

FEATURES = FeatureConfig(n_mels=12)
SAMPLES = (
    generate_dataset(
        GenerationConfig(
            domain="weather", seed=101, counts={"train": 6}, n_speakers=2, noise_snr_db=(40.0, 15.0)
        )
    )
    .examples[2]
    .utterance.samples
)
HALF = (len(SAMPLES) // 2) // 640 * 640

GOLDEN_STATE_KEYS = [
    "buffer",
    "committed_tags",
    "feature_config",
    "finished",
    "first_frame_ms",
    "first_stable_ms",
    "frame_cursor",
    "hidden",
    "mel_window",
    "model_hash",
    "n_pushed",
    "revisions",
    "schema",
    "schema_version",
    "stable_intent",
    "state_count",
    "state_sum",
    "stream_config",
]


def make_model(encoder="tcn", hidden_size=16):
    torch.manual_seed(4242)
    config = TrainConfig(encoder=encoder, hidden_size=hidden_size, layers=2)
    model = JointIntentSlotModel(ModelSpec.from_domain(weather_domain(), FEATURES, config))
    model.eval()
    return model


def streamed(model, samples, start=0):
    session = StreamSession(model, FEATURES, StreamConfig())
    for offset in range(start, len(samples), 640):
        session.push_audio(samples[offset : offset + 640])
    return session


@pytest.mark.parametrize("encoder", ["tcn", "gru"])
def test_resume_matches_uninterrupted(encoder):
    model = make_model(encoder)
    full = streamed(model, SAMPLES).finalize()
    first = streamed(model, SAMPLES[:HALF])
    second = StreamSession.from_dict(first.to_dict(), model)
    for offset in range(HALF, len(SAMPLES), 640):
        second.push_audio(SAMPLES[offset : offset + 640])
    assert second.finalize() == full


def test_state_keys_pinned_and_json_ready():
    model = make_model()
    state = streamed(model, SAMPLES[:HALF]).to_dict()
    assert sorted(state) == GOLDEN_STATE_KEYS
    assert isinstance(stable_hash(state), str)


def test_state_strict_validation():
    model = make_model()
    state = streamed(model, SAMPLES[:HALF]).to_dict()
    with pytest.raises(StreamingError, match="unknown keys"):
        StreamSession.from_dict({**state, "extra": 1}, model)
    with pytest.raises(StreamingError, match="missing keys"):
        StreamSession.from_dict({k: v for k, v in state.items() if k != "buffer"}, model)
    with pytest.raises(StreamingError, match="schema"):
        StreamSession.from_dict({**state, "schema": "other"}, model)
    with pytest.raises(StreamingError, match="version"):
        StreamSession.from_dict({**state, "schema_version": 2}, model)
    with pytest.raises(StreamingError, match="must be a dict"):
        StreamSession.from_dict(["state"], model)


def test_tampered_counters_rejected():
    model = make_model()
    state = streamed(model, SAMPLES[:HALF]).to_dict()
    with pytest.raises(StreamingError, match="counters disagree"):
        StreamSession.from_dict({**state, "frame_cursor": state["frame_cursor"] + 5}, model)
    with pytest.raises(StreamingError, match="behind"):
        StreamSession.from_dict({**state, "n_pushed": 0}, model)


def test_foreign_model_state_rejected():
    model = make_model()
    other = make_model(hidden_size=8)
    state = streamed(model, SAMPLES[:HALF]).to_dict()
    with pytest.raises(StreamingError, match="different model"):
        StreamSession.from_dict(state, other)
