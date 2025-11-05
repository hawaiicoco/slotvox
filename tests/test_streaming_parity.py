"""Offline equivalence of committed frame tags (greedy path).

The central streaming invariant: for greedy decoding, tags committed by a
chunked session equal the offline frame tags exactly — for the TCN (whose
bounded context window covers the receptive field) and for the GRU
(carried hidden state), at every chunk size tested.
"""

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


def make_model(encoder):
    torch.manual_seed(4242)
    config = TrainConfig(encoder=encoder, hidden_size=16, layers=2)
    model = JointIntentSlotModel(ModelSpec.from_domain(weather_domain(), FEATURES, config))
    model.eval()
    return model


def utterance_samples():
    dataset = generate_dataset(
        GenerationConfig(
            domain="weather",
            seed=101,
            counts={"train": 6},
            n_speakers=2,
            noise_snr_db=(40.0, 15.0),
        )
    )
    return [example.utterance.samples for example in dataset.examples[:3]]


def offline_tags(model, samples):
    with torch.no_grad():
        mel = torch.from_numpy(log_mel(samples, FEATURES)).unsqueeze(0)
        _, slot_logits = model(mel)
    names = tuple(model.spec.tags[i] for i in slot_logits.argmax(dim=-1)[0].tolist())
    return names, mel.shape[1]


@pytest.mark.parametrize("encoder", ["tcn", "gru"])
@pytest.mark.parametrize("chunk_ms", [160, 37, 10])
def test_committed_tags_equal_offline(encoder, chunk_ms):
    model = make_model(encoder)
    for samples in utterance_samples():
        expected, n_frames = offline_tags(model, samples)
        session = StreamSession(model, FEATURES, StreamConfig(chunk_ms=chunk_ms))
        chunk = FEATURES.sample_rate * chunk_ms // 1000
        for start in range(0, len(samples), chunk):
            session.push_audio(samples[start : start + chunk])
        result = session.finalize()
        assert result.frame_tags == expected
        assert result.n_frames == n_frames
