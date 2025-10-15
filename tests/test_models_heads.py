"""Intent/slot head contracts, incl. streaming accumulation parity."""

import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.model]

from slotvox.errors import ValidationError  # noqa: E402
from slotvox.models.heads import IntentHead, SlotHead  # noqa: E402


def test_masked_pooling_matches_manual_mean():
    torch.manual_seed(3)
    head = IntentHead(6, 3).eval()
    states = torch.randn(2, 5, 6)
    mask = torch.tensor([[True] * 5, [True, True, True, False, False]])
    pooled = torch.stack([states[0].mean(dim=0), states[1][:3].mean(dim=0)])
    with torch.no_grad():
        assert torch.allclose(head(states, mask), head.dense(pooled), atol=1e-6)
        assert head(states, mask).shape == (2, 3)


def test_unmasked_forward_is_plain_mean():
    torch.manual_seed(4)
    head = IntentHead(5, 2).eval()
    states = torch.randn(1, 4, 5)
    with torch.no_grad():
        assert torch.allclose(head(states), head.dense(states.mean(dim=1)), atol=1e-6)


def test_accumulated_parity_with_forward():
    torch.manual_seed(5)
    head = IntentHead(5, 2).eval()
    states = torch.randn(2, 6, 5)
    counts = torch.tensor([6, 4])
    mask = torch.tensor([[True] * 6, [True] * 4 + [False, False]])
    with torch.no_grad():
        offline = head(states, mask)
        streamed = head.from_accumulated((states * mask.unsqueeze(-1)).sum(dim=1), counts)
    assert torch.allclose(offline, streamed, atol=1e-6)


def test_zero_count_does_not_produce_nan():
    head = IntentHead(3, 2)
    with torch.no_grad():
        logits = head.from_accumulated(torch.zeros(1, 3), torch.tensor([0]))
    assert torch.all(torch.isfinite(logits))


def test_input_validation():
    head = IntentHead(4, 2)
    with pytest.raises(ValidationError):
        head(torch.randn(4, 4))
    with pytest.raises(ValidationError):
        head(torch.randn(1, 3, 4), torch.ones(1, 3))  # float mask
    with pytest.raises(ValidationError):
        head(torch.randn(1, 3, 4), torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(ValidationError):
        head.from_accumulated(torch.zeros(3), torch.tensor([1]))
    with pytest.raises(ValidationError):
        SlotHead(4, 0)
    with pytest.raises(ValidationError):
        SlotHead(4, 3)(torch.randn(3, 4))
