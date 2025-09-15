"""frame_token_map / frame_tags contracts."""

import numpy as np
import pytest

from slotvox.config import FeatureConfig
from slotvox.data.alignment import NO_TOKEN, frame_tags, frame_token_map
from slotvox.errors import ValidationError
from slotvox.features.frontend import log_mel_frames
from slotvox.synth.utterance import SegmentPlan, render_utterance

PLANS = [
    SegmentPlan("a", "tone", 80, {"freq": 300.0}),
    SegmentPlan("b", "tone", 80, {"freq": 500.0}),
]
UTT = render_utterance(PLANS, 16000, gap_ms=20)
CFG = FeatureConfig()


def test_length_matches_log_mel_frames():
    token_map = frame_token_map(UTT, CFG)
    assert token_map.shape == (log_mel_frames(UTT.n_samples, CFG),)
    assert token_map.dtype == np.int64


def test_gaps_map_to_no_token_and_segments_to_indices():
    token_map = frame_token_map(UTT, CFG)
    assert token_map[0] == NO_TOKEN  # first frame center lands in the lead gap
    assert token_map[-1] == NO_TOKEN  # trailing tail is past the last segment
    assert set(np.unique(token_map)) <= {NO_TOKEN, 0, 1}
    assert 0 in token_map
    assert 1 in token_map


def test_token_indices_never_decrease():
    present = frame_token_map(UTT, CFG)
    present = present[present != NO_TOKEN]
    assert np.all(np.diff(present) >= 0)


def test_frame_tags_lifts_and_pads():
    token_map = frame_token_map(UTT, CFG)
    lifted = frame_tags(("B-x", "I-y"), token_map)
    assert len(lifted) == len(token_map)
    assert lifted[0] == "O"
    first_token_frame = int(np.flatnonzero(token_map == 1)[0])
    assert lifted[first_token_frame] == "I-y"
    assert set(lifted) <= {"O", "B-x", "I-y"}


def test_frame_tags_rejects_bad_inputs():
    with pytest.raises(ValidationError, match="within"):
        frame_tags(("O",), np.array([1], dtype=np.int64))
    with pytest.raises(ValidationError, match="within"):
        frame_tags(("O",), np.array([-2], dtype=np.int64))
    with pytest.raises(ValidationError, match="integer dtype"):
        frame_tags(("O",), np.array([0.5]))
    with pytest.raises(ValidationError):
        frame_tags("O", np.array([0], dtype=np.int64))
    with pytest.raises(ValidationError):
        frame_tags((1,), np.array([0], dtype=np.int64))


def test_sample_rate_mismatch_rejected():
    config = FeatureConfig(sample_rate=8000, fmax=3900.0)
    with pytest.raises(ValidationError, match="sample rate mismatch"):
        frame_token_map(UTT, config)


def test_type_rejections():
    with pytest.raises(ValidationError):
        frame_token_map("not-an-utterance", CFG)
    with pytest.raises(ValidationError):
        frame_token_map(UTT, "not-a-config")
