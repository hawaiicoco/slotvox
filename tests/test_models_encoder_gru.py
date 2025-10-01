"""GRU encoder and build_encoder contracts."""

import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.model]

from slotvox.config import TrainConfig  # noqa: E402
from slotvox.errors import ValidationError  # noqa: E402
from slotvox.models.encoder import GRUEncoder, TCNEncoder, build_encoder  # noqa: E402


def test_gru_is_causal():
    torch.manual_seed(1)
    encoder = GRUEncoder(4, 8, 1, 0.0).eval()
    x = torch.randn(1, 10, 4)
    perturbed = x.clone()
    perturbed[:, 6:] += 5.0
    with torch.no_grad():
        base, _ = encoder(x)
        after, _ = encoder(perturbed)
    assert torch.allclose(base[:, :6], after[:, :6], atol=1e-6)
    assert encoder.receptive_field is None


def test_gru_hidden_state_continues_exactly():
    torch.manual_seed(2)
    encoder = GRUEncoder(4, 8, 1, 0.0).eval()
    x = torch.randn(1, 8, 4)
    with torch.no_grad():
        whole, _ = encoder(x)
        first, hidden = encoder(x[:, :4])
        second, _ = encoder(x[:, 4:], hidden)
    assert torch.allclose(torch.cat([first, second], dim=1), whole, atol=1e-6)


def test_factory_builds_configured_encoder():
    tcn = build_encoder(TrainConfig(encoder="tcn"), 6)
    gru = build_encoder(TrainConfig(encoder="gru"), 6)
    assert isinstance(tcn, TCNEncoder)
    assert isinstance(gru, GRUEncoder)


def test_factory_rejects_bad_inputs():
    with pytest.raises(ValidationError, match="TrainConfig"):
        build_encoder("tcn", 6)
    config = TrainConfig()
    object.__setattr__(config, "encoder", "lstm")
    with pytest.raises(ValidationError, match="unknown encoder"):
        build_encoder(config, 6)
    with pytest.raises(ValidationError):
        GRUEncoder(4, 8, 1, 1.5)
    with pytest.raises(ValidationError):
        GRUEncoder(4, 8, 1, 0.0)(torch.randn(4, 4))
