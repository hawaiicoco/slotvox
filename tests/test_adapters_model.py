"""JointAdapter: real-model inference over the protocol (synthetic audio)."""

import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.model]

import numpy as np  # noqa: E402

from slotvox.adapters.model import JointAdapter  # noqa: E402
from slotvox.adapters.protocol import InferRequest  # noqa: E402
from slotvox.config import FeatureConfig, GenerationConfig, TrainConfig  # noqa: E402
from slotvox.data.dataset import generate_dataset  # noqa: E402
from slotvox.errors import AdapterError  # noqa: E402
from slotvox.features.frontend import log_mel  # noqa: E402
from slotvox.models.joint import JointIntentSlotModel, ModelSpec  # noqa: E402
from slotvox.schema.builtins import weather_domain  # noqa: E402
from slotvox.tagging.bio import repair_sequence  # noqa: E402

FEATURES = FeatureConfig(n_mels=12)


def make_model(encoder="tcn"):
    torch.manual_seed(77)
    spec = ModelSpec.from_domain(
        weather_domain(), FEATURES, TrainConfig(encoder=encoder, hidden_size=8, layers=1)
    )
    model = JointIntentSlotModel(spec)
    model.eval()
    return model


def make_request():
    data = generate_dataset(
        GenerationConfig(domain="weather", seed=41, counts={"train": 1}, n_speakers=1)
    )
    samples = data.examples[0].utterance.samples
    return InferRequest("req-model", tuple(float(value) for value in samples), 16000)


def test_infer_returns_protocol_valid_response():
    adapter = JointAdapter(make_model())
    response = adapter.infer(make_request())
    assert response.request_id == "req-model"
    assert response.intent in weather_domain().intent_names
    assert 0.0 <= response.posterior <= 1.0
    assert response.model_hash == adapter.model_hash
    assert len(response.frame_tags) > 0


@pytest.mark.parametrize("encoder", ["tcn", "gru"])
def test_deterministic_and_greedy_consistent(encoder):
    model = make_model(encoder)
    adapter = JointAdapter(model)
    request = make_request()
    first = adapter.infer(request)
    assert adapter.infer(request) == first
    mel = log_mel(np.asarray(request.samples, dtype=np.float64), FEATURES)
    intent, tags = model.predict_greedy(torch.from_numpy(mel))
    assert first.intent == intent
    assert first.frame_tags == repair_sequence(tags, "promote")


def test_rejections():
    adapter = JointAdapter(make_model())
    with pytest.raises(AdapterError, match="sample rate"):
        adapter.infer(InferRequest("r", (0.1, 0.2), 8000))
    with pytest.raises(AdapterError, match="analysis frame"):
        adapter.infer(InferRequest("r", tuple([0.01] * 100), 16000))
    with pytest.raises(AdapterError, match="InferRequest"):
        adapter.infer("nope")
    with pytest.raises(AdapterError, match="JointIntentSlotModel"):
        JointAdapter(object())
