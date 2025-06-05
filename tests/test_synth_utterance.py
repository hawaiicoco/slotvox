"""SyntheticUtterance validation, views, and metadata roundtrip."""

import numpy as np
import pytest

from slotvox.audio.wav import WavAudio
from slotvox.errors import ValidationError
from slotvox.synth.segments import Segment
from slotvox.synth.utterance import SyntheticUtterance


def audio(n=1000):
    return WavAudio(np.zeros(n, dtype=np.float32), 16000)


def seg(start, end, label="x", freq=440.0):
    return Segment(
        label=label, kind="tone", start_sample=start, end_sample=end, params={"freq": freq}
    )


def utterance(segments, n=1000, snr=None):
    return SyntheticUtterance(audio=audio(n), segments=tuple(segments), snr_db=snr)


def test_geometry_properties():
    utt = utterance([seg(0, 400), seg(400, 900)])
    assert utt.n_samples == 1000
    assert utt.sample_rate == 16000
    assert utt.duration_s == pytest.approx(1000 / 16000)


def test_overlapping_or_unsorted_segments_rejected():
    with pytest.raises(ValidationError, match="non-overlapping"):
        utterance([seg(0, 500), seg(300, 700)])
    with pytest.raises(ValidationError, match="non-overlapping"):
        utterance([seg(500, 700), seg(0, 400)])


def test_out_of_bounds_segment_rejected():
    with pytest.raises(ValidationError, match="exceeds utterance bounds"):
        utterance([seg(0, 1001)])


def test_foreign_segment_view_rejected():
    utt = utterance([seg(0, 400)])
    with pytest.raises(ValidationError, match="does not belong"):
        utt.segment_view(seg(0, 400, label="other"))


def test_metadata_round_trip_and_strict_keys():
    utt = utterance([seg(0, 400, label="intent:hi")], snr=12.5)
    payload = utt.to_dict()
    rebuilt = SyntheticUtterance.from_dict(payload, np.zeros(1000, dtype=np.float32))
    assert rebuilt == utt
    with pytest.raises(ValidationError, match="unknown keys"):
        SyntheticUtterance.from_dict({**payload, "junk": 1}, np.zeros(1000, dtype=np.float32))
    with pytest.raises(ValidationError, match="n_samples"):
        SyntheticUtterance.from_dict(payload, np.zeros(999, dtype=np.float32))


def test_invalid_snr_rejected():
    with pytest.raises(ValidationError, match="snr_db"):
        utterance([seg(0, 400)], snr=float("nan"))
    with pytest.raises(ValidationError, match="snr_db"):
        utterance([seg(0, 400)], snr="loud")
