"""Frame commit geometry: lookahead, flush, boundaries, lifecycle errors."""

import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.model]

from slotvox.config import FeatureConfig, GenerationConfig, StreamConfig, TrainConfig  # noqa: E402
from slotvox.data.dataset import generate_dataset  # noqa: E402
from slotvox.errors import StreamingError  # noqa: E402
from slotvox.features.frontend import log_mel_frames  # noqa: E402
from slotvox.models.joint import JointIntentSlotModel, ModelSpec  # noqa: E402
from slotvox.schema.builtins import weather_domain  # noqa: E402
from slotvox.streaming.session import StreamSession  # noqa: E402

FEATURES = FeatureConfig(n_mels=12)

torch.manual_seed(4242)
MODEL = JointIntentSlotModel(
    ModelSpec.from_domain(
        weather_domain(), FEATURES, TrainConfig(encoder="tcn", hidden_size=16, layers=2)
    )
)
MODEL.eval()
SAMPLES = (
    generate_dataset(
        GenerationConfig(
            domain="weather", seed=101, counts={"train": 6}, n_speakers=2, noise_snr_db=(40.0, 15.0)
        )
    )
    .examples[2]
    .utterance.samples
)


def session(**kwargs):
    return StreamSession(MODEL, FEATURES, StreamConfig(**kwargs))


def push_all(target, chunk=640):
    for start in range(0, len(SAMPLES), chunk):
        target.push_audio(SAMPLES[start : start + chunk])


def test_frame_count_matches_offline_at_every_chunk_size():
    for chunk in (640, 592, 160):
        current = session()
        push_all(current, chunk)
        result = current.finalize()
        assert result.n_frames == log_mel_frames(len(SAMPLES), FEATURES) == 80


def test_nothing_commits_before_lookahead_arrives():
    current = session()
    assert current.push_audio(SAMPLES[:400]) == 0
    assert current.frames_committed == 0
    current.push_audio(SAMPLES[400:5760])
    assert current.frames_committed == 2  # (5760 - 5120 - 400) // 160 + 1


def test_finalize_flushes_frames_without_lookahead():
    current = session()
    current.push_audio(SAMPLES[:5000])
    assert current.frames_committed == 0
    result = current.finalize()
    assert result.n_frames == (5000 - 400) // 160 + 1 == 29


def test_short_stream_finalize_raises():
    current = session()
    current.push_audio(SAMPLES[:100])
    with pytest.raises(StreamingError, match="shorter than one analysis frame"):
        current.finalize()


def test_lifecycle_errors():
    current = session()
    push_all(current)
    current.finalize()
    with pytest.raises(StreamingError, match="finished"):
        current.push_audio(SAMPLES[:100])
    with pytest.raises(StreamingError, match="already finalized"):
        current.finalize()


def test_init_validation():
    torch.manual_seed(1)
    training = JointIntentSlotModel(
        ModelSpec.from_domain(weather_domain(), FEATURES, TrainConfig(hidden_size=8, layers=1))
    )
    training.train()
    with pytest.raises(StreamingError, match="eval mode"):
        StreamSession(training, FEATURES, StreamConfig())
    with pytest.raises(StreamingError, match="sample rate mismatch"):
        StreamSession(MODEL, FEATURES, StreamConfig(sample_rate=8000))
    with pytest.raises(StreamingError, match="JointIntentSlotModel"):
        StreamSession("model", FEATURES, StreamConfig())
    with pytest.raises(StreamingError, match="FeatureConfig"):
        StreamSession(MODEL, "cfg", StreamConfig())
