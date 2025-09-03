"""GeneratedExample cross-object consistency contracts."""

from dataclasses import replace

import numpy as np
import pytest

from slotvox.audio.wav import WavAudio
from slotvox.data.factory import GeneratedExample
from slotvox.errors import ValidationError
from slotvox.schema.annotations import AnnotatedUtterance
from slotvox.synth.segments import Segment
from slotvox.synth.utterance import SyntheticUtterance

TOKENS = ("你", "好", "北", "京")
TAGS = ("O", "O", "B-city", "I-city")
SEGMENTS = (
    Segment("lit:你", "tone", 0, 100, {"freq": 200.0}),
    Segment("lit:好", "tone", 100, 200, {"freq": 210.0}),
    Segment("city:北", "formant", 200, 300, {"formants": [500.0, 1200.0]}),
    Segment("city:京", "formant", 300, 400, {"formants": [500.0, 1250.0]}),
)


def build(**overrides):
    annotation = AnnotatedUtterance("utt-1", "zh", TOKENS, TAGS, "query-weather")
    utterance = SyntheticUtterance(WavAudio(np.zeros(400, dtype=np.float32), 16000), SEGMENTS, None)
    data = {
        "utterance_id": "utt-1",
        "split": "train",
        "speaker_id": 0,
        "snr_db": None,
        "pattern_id": "weather/query-weather/0",
        "annotation": annotation,
        "utterance": utterance,
        "slot_values": {"city": ("北京",)},
    }
    data.update(overrides)
    return GeneratedExample(**data)


def test_valid_example_and_manifest_keys():
    row = build().to_dict()
    assert row["utterance_id"] == "utt-1"
    assert len(row["segments"]) == 4
    assert sorted(row) == [
        "annotation",
        "n_samples",
        "pattern_id",
        "sample_rate",
        "segments",
        "slot_values",
        "snr_db",
        "speaker_id",
        "split",
        "utterance_id",
    ]


def test_slot_values_lists_normalize_to_tuples():
    assert build(slot_values={"city": ["北京"]}).slot_values == {"city": ("北京",)}


def test_misaligned_segments_rejected():
    short = SyntheticUtterance(WavAudio(np.zeros(400, dtype=np.float32), 16000), SEGMENTS[:3], None)
    with pytest.raises(ValidationError, match="misaligned"):
        build(utterance=short)


def test_id_and_snr_mismatches_rejected():
    other = AnnotatedUtterance("utt-2", "zh", TOKENS, TAGS, "query-weather")
    with pytest.raises(ValidationError, match="utterance_id"):
        build(annotation=other)
    with pytest.raises(ValidationError, match="snr_db"):
        build(snr_db=10.0)


def test_slot_value_vocab_mismatches_rejected():
    with pytest.raises(ValidationError, match="tagged slot names"):
        build(slot_values={"day": ("明天",)})
    with pytest.raises(ValidationError, match="tagged spans"):
        build(slot_values={"city": ("北京", "上海")})


def test_field_validation():
    with pytest.raises(ValidationError):
        build(utterance_id="")
    with pytest.raises(ValidationError):
        build(utterance_id="has space")
    with pytest.raises(ValidationError):
        build(split="nope")
    with pytest.raises(ValidationError):
        build(speaker_id=-1)
    with pytest.raises(ValidationError):
        build(snr_db="loud")
    with pytest.raises(ValidationError):
        build(pattern_id="")


def test_replace_reruns_validation():
    with pytest.raises(ValidationError, match="snr_db"):
        replace(build(), snr_db=5.0)
