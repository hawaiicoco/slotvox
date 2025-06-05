"""Segment dataclass and render-dispatch contracts."""

import numpy as np
import pytest

from slotvox.errors import ValidationError
from slotvox.synth.segments import SIGNAL_KINDS, Segment, render_segment, validate_params
from slotvox.synth.signals import tone


def make_segment(**overrides):
    data = {
        "label": "slot:city",
        "kind": "tone",
        "start_sample": 0,
        "end_sample": 160,
        "params": {"freq": 440.0},
    }
    data.update(overrides)
    return Segment(**data)


def test_round_trip_through_dict():
    segment = make_segment()
    assert Segment.from_dict(segment.to_dict()) == segment


def test_from_dict_strict_keys():
    payload = make_segment().to_dict()
    with pytest.raises(ValidationError, match="unknown keys"):
        Segment.from_dict({**payload, "extra": 1})
    missing = dict(payload)
    del missing["label"]
    with pytest.raises(ValidationError, match="missing keys"):
        Segment.from_dict(missing)
    with pytest.raises(ValidationError, match="must be a dict"):
        Segment.from_dict(["not", "a", "dict"])


@pytest.mark.parametrize(
    "overrides",
    [
        {"label": ""},
        {"kind": "noise"},
        {"start_sample": -1},
        {"end_sample": 0},
        {"params": {}},
        {"params": {"freq": 440.0, "gain": 2}},
        {"params": {"freq": "440"}},
    ],
)
def test_invalid_segments_rejected(overrides):
    with pytest.raises(ValidationError):
        make_segment(**overrides)


def test_validate_params_covers_every_kind_both_directions():
    good = {
        "tone": {"freq": 440.0},
        "chirp": {"f_start": 400.0, "f_end": 900.0},
        "formant": {"formants": [300.0, 900.0]},
    }
    assert set(good) == set(SIGNAL_KINDS)
    for kind, params in good.items():
        validate_params(kind, params)
    with pytest.raises(ValidationError, match="f_start < f_end"):
        validate_params("chirp", {"f_start": 900.0, "f_end": 400.0})
    with pytest.raises(ValidationError, match="strictly ascending"):
        validate_params("formant", {"formants": [900.0, 300.0]})


def test_params_must_be_json_native():
    with pytest.raises(ValidationError):
        validate_params("tone", {"freq": np.float32(440.0)})


def test_render_segment_dispatches_per_kind():
    rendered = render_segment("tone", {"freq": 440.0}, 160, 16000)
    assert rendered.dtype == np.float32
    assert rendered.shape == (160,)
    assert np.array_equal(rendered, tone(440.0, 160, 16000, amplitude=0.8))
    for kind, params in [
        ("chirp", {"f_start": 400.0, "f_end": 900.0}),
        ("formant", {"formants": [300.0, 900.0]}),
    ]:
        assert render_segment(kind, params, 200, 16000).shape == (200,)
    with pytest.raises(ValidationError):
        render_segment("noise", {}, 160, 16000)
