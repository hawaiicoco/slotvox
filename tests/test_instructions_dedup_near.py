"""Feature-hash near-duplicate detection with measured thresholds.

Similarity regimes below were measured on these exact fixtures: a single
slot-value edit sits at ~0.95, while cross-language and cross-domain
pairs sit at ~0.37-0.46 — the 0.9 default threshold separates them with
a wide margin.
"""

import numpy as np
import pytest

from slotvox.errors import ValidationError
from slotvox.instructions.dedup import (
    FEATURE_WIDTH,
    NEAR_DUPLICATE_THRESHOLD,
    cosine_similarity,
    feature_vector,
    near_duplicate_pairs,
)
from slotvox.instructions.schema import AudioRef, InstructionSample, Turn

HASH = "c3" * 32
BASE = (
    "你是语音助手。听用户的语音，识别weather领域的意图和槽位。\n"
    "意图：query-weather；槽位：city=上海, day=今天"
)
EDIT = BASE.replace("上海", "北京")
DISTANT_EN = (
    "You are a speech assistant. Listen to the user's audio and identify "
    "the intent and slots for the weather domain.\n"
    "Intent: query-forecast; Slots: city=Beijing, day=tomorrow"
)
DISTANT_ZH = (
    "你是语音助手。听用户的语音，识别music-control领域的意图和槽位。\n"
    "意图：play-song；槽位：artist=周杰伦, song=晴天"
)


def make_sample(sample_id, text):
    return InstructionSample(
        sample_id=sample_id,
        domain="weather",
        language="zh",
        intent="query-weather",
        turns=(
            Turn("user", audio=AudioRef(sample_id, HASH, 300.0)),
            Turn("assistant", text=text),
        ),
    )


def test_feature_vector_shape_norm_and_determinism():
    vector = feature_vector(BASE)
    assert vector.shape == (FEATURE_WIDTH,)
    assert np.isclose(np.linalg.norm(vector), 1.0)
    assert np.array_equal(vector, feature_vector(BASE))
    assert not feature_vector("！！！").any()  # normalizes to nothing -> zero vector


def test_measured_similarity_regimes():
    assert cosine_similarity(feature_vector(BASE), feature_vector(EDIT)) > 0.94
    assert cosine_similarity(feature_vector(BASE), feature_vector(DISTANT_EN)) < 0.5
    assert cosine_similarity(feature_vector(BASE), feature_vector(DISTANT_ZH)) < 0.5
    assert cosine_similarity(np.zeros(4), np.ones(4)) == 0.0


def test_near_pairs_default_threshold_finds_only_the_edit():
    samples = [
        make_sample("s-base", BASE),
        make_sample("s-edit", EDIT),
        make_sample("s-en", DISTANT_EN),
        make_sample("s-music", DISTANT_ZH),
    ]
    pairs = near_duplicate_pairs(samples)
    assert [(left, right) for left, right, _ in pairs] == [("s-base", "s-edit")]
    assert pairs[0][2] >= NEAR_DUPLICATE_THRESHOLD


def test_threshold_validation_and_shapes():
    with pytest.raises(ValidationError, match="threshold"):
        near_duplicate_pairs([], threshold=1.5)
    with pytest.raises(ValidationError, match="threshold"):
        near_duplicate_pairs([], threshold=-0.1)
    with pytest.raises(ValidationError, match="mismatch"):
        cosine_similarity(np.zeros(3), np.zeros(4))
    with pytest.raises(ValidationError, match="1-D"):
        cosine_similarity(np.zeros((2, 2)), np.zeros((2, 2)))
