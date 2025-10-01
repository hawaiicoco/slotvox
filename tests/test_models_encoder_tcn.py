"""TCN encoder contracts: shapes, strict causality, exact receptive field."""

import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.model]

from slotvox.errors import ValidationError  # noqa: E402
from slotvox.models.encoder import TCNEncoder  # noqa: E402


def make_encoder(layers=2, hidden=8):
    torch.manual_seed(0)
    return TCNEncoder(4, hidden, layers, 0.0).eval()


def test_output_shape():
    encoder = make_encoder()
    with torch.no_grad():
        out = encoder(torch.randn(2, 12, 4))
    assert out.shape == (2, 12, 8)


def test_strictly_causal_prefixes_are_bit_identical():
    encoder = make_encoder()
    x = torch.randn(1, 12, 4)
    perturbed = x.clone()
    perturbed[:, 7:] += 10.0
    with torch.no_grad():
        base, after = encoder(x), encoder(perturbed)
    assert torch.equal(base[:, :7], after[:, :7])


def test_receptive_field_is_exact():
    encoder = make_encoder(layers=2)
    assert encoder.receptive_field == 7  # 1 + 2*1 + 2*2
    deep = make_encoder(layers=3)
    assert deep.receptive_field == 15  # 1 + 2*(1 + 2 + 4)


def test_frames_beyond_receptive_field_are_unaffected():
    encoder = make_encoder(layers=2)  # receptive field 7
    x = torch.randn(1, 16, 4)
    far = x.clone()
    far[:, :5] += 10.0
    with torch.no_grad():
        base, after = encoder(x), encoder(far)
    # frame j sees [j-6, j]; frames >= 11 cannot see frames < 5
    assert torch.equal(base[:, 11:], after[:, 11:])
    assert not torch.equal(base[:, :11], after[:, :11])


def test_invalid_construction_and_inputs():
    with pytest.raises(ValidationError):
        TCNEncoder(0, 8, 1, 0.0)
    with pytest.raises(ValidationError):
        TCNEncoder(4, 8, 0, 0.0)
    with pytest.raises(ValidationError):
        TCNEncoder(4, 8, 2, 1.0)
    with pytest.raises(ValidationError):
        make_encoder()(torch.randn(5, 4))
