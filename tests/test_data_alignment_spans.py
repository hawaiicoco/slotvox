"""frame_span_tags contracts: canonical spans, validity property, goldens."""

import numpy as np
import pytest

from slotvox.config import FeatureConfig, GenerationConfig
from slotvox.data.alignment import frame_span_tags, frame_token_map
from slotvox.data.dataset import generate_dataset
from slotvox.errors import TaggingError, ValidationError
from slotvox.tagging.bio import is_valid_sequence, tags_to_spans


def test_gap_inside_span_continues_the_span():
    token_map = np.array([-1, 0, 0, -1, 1, 1, -1], dtype=np.int64)
    lifted = frame_span_tags(("B-city", "I-city"), token_map)
    assert lifted == (
        "O",
        "B-city",
        "I-city",
        "I-city",
        "I-city",
        "I-city",
        "O",
    )


def test_spans_without_frame_coverage_are_dropped():
    assert frame_span_tags(("B-x",), np.array([-1, -1], dtype=np.int64)) == ("O", "O")
    assert frame_span_tags((), np.zeros(0, dtype=np.int64)) == ()


def test_structurally_invalid_token_tags_are_rejected():
    with pytest.raises(TaggingError):
        frame_span_tags(("I-city",), np.array([0], dtype=np.int64))


def test_input_validation():
    with pytest.raises(ValidationError):
        frame_span_tags("B-x", np.array([0], dtype=np.int64))
    with pytest.raises(ValidationError):
        frame_span_tags(("B-x",), np.array([0.0]))
    with pytest.raises(ValidationError):
        frame_span_tags(("B-x",), np.array([5], dtype=np.int64))
    with pytest.raises(ValidationError):
        frame_span_tags((1,), np.array([0], dtype=np.int64))


def test_factory_frames_are_always_valid_bio_with_matching_labels():
    dataset = generate_dataset(
        GenerationConfig(domain="weather", seed=91, counts={"train": 12}, n_speakers=2)
    )
    config = FeatureConfig(n_mels=8)
    checked = 0
    for example in dataset.examples:
        token_map = frame_token_map(example.utterance, config)
        lifted = frame_span_tags(example.annotation.tags, token_map)
        assert is_valid_sequence(lifted)
        token_spans = tags_to_spans(example.annotation.tags)
        frame_spans = tags_to_spans(lifted)
        assert [span.label for span in frame_spans] == [span.label for span in token_spans]
        checked += len(frame_spans)
    assert checked > 0
