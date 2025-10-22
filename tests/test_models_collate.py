"""collate/iterate_batches contracts: padding, masks, determinism."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.model]

from slotvox.errors import ValidationError  # noqa: E402
from slotvox.models.dataset import EncodedExample, collate, iterate_batches  # noqa: E402
from slotvox.models.vocab import PAD_INDEX  # noqa: E402
from slotvox.util.seed import make_rng  # noqa: E402


def example(utterance_id, frames, features=3, intent=0):
    return EncodedExample(
        utterance_id=utterance_id,
        mel=np.ones((frames, features), dtype=np.float32),
        frame_tag_indices=np.arange(frames, dtype=np.int64) % 5,
        intent_index=intent,
        frame_tag_names=tuple("O" for _ in range(frames)),
    )


def test_collate_pads_and_masks():
    batch = collate([example("a", 5), example("b", 3)])
    assert tuple(batch.mel.shape) == (2, 5, 3)
    assert batch.mask.tolist() == [[True] * 5, [True] * 3 + [False] * 2]
    assert torch.all(batch.mel[1, 3:] == 0.0)
    assert torch.all(batch.tags[1, 3:] == PAD_INDEX)
    assert batch.size == 2


def test_collate_rejections():
    with pytest.raises(ValidationError):
        collate([])
    with pytest.raises(ValidationError):
        collate(["not-an-example"])
    with pytest.raises(ValidationError, match="feature width"):
        collate([example("a", 4, features=3), example("b", 4, features=5)])


def test_iterate_batches_sequential_covers_everything():
    examples = [example(f"e{i}", 4) for i in range(5)]
    batches = list(iterate_batches(examples, 2))
    assert [batch.size for batch in batches] == [2, 2, 1]


def test_iterate_batches_shuffled_is_deterministic_per_seed():
    examples = [example(f"e{i}", 4, intent=i) for i in range(6)]
    first = [b.intent.tolist() for b in iterate_batches(examples, 3, rng=make_rng(9))]
    again = [b.intent.tolist() for b in iterate_batches(examples, 3, rng=make_rng(9))]
    other = [b.intent.tolist() for b in iterate_batches(examples, 3, rng=make_rng(10))]
    assert first == again
    assert sorted(sum(first, [])) == sorted(sum(other, [])) == list(range(6))


def test_iterate_batches_validation():
    examples = [example("a", 4)]
    for bad in (0, -1, True, 2.0):
        with pytest.raises(ValidationError):
            list(iterate_batches(examples, bad))
    with pytest.raises(ValidationError):
        list(iterate_batches([], 2))
    with pytest.raises(ValidationError):
        list(iterate_batches(examples, 2, rng=7))


def test_encoded_example_validation():
    with pytest.raises(ValidationError):
        EncodedExample("x", np.zeros((3, 2), np.float32), np.zeros(2, np.int64), 0, ("O",) * 3)
    with pytest.raises(ValidationError):
        EncodedExample("x", np.zeros((3, 2), np.float64), np.zeros(3, np.int64), 0, ("O",) * 3)
    with pytest.raises(ValidationError):
        EncodedExample("x", np.zeros((3, 2), np.float32), np.zeros(3, np.int64), -1, ("O",) * 3)
