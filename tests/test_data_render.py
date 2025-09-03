"""render_example contracts: alignment by construction, determinism."""

import numpy as np

from slotvox.data.factory import render_example
from slotvox.data.patterns import templates_for
from slotvox.schema.builtins import weather_domain

DOMAIN = weather_domain()
TEMPLATE = templates_for("weather", "zh")[0]


def render(speaker_id=0, snr_db=None, seed=9):
    return render_example(
        DOMAIN,
        TEMPLATE,
        "zh",
        utterance_id="utt-x",
        split="train",
        speaker_id=speaker_id,
        snr_db=snr_db,
        seed=seed,
    )


def test_alignment_by_construction():
    example = render()
    assert len(example.utterance.segments) == len(example.annotation.tokens)
    for segment in example.utterance.segments:
        view = example.utterance.segment_view(segment)
        assert float(np.max(np.abs(view))) > 0.0
    example.annotation.validate_against(DOMAIN)


def test_rendering_is_deterministic():
    assert np.array_equal(render(seed=11).utterance.samples, render(seed=11).utterance.samples)


def test_noise_changes_audio_not_labels():
    clean = render(seed=13)
    noisy = render(seed=13, snr_db=15.0)
    assert clean.annotation == noisy.annotation
    assert not np.array_equal(clean.utterance.samples, noisy.utterance.samples)
    assert noisy.snr_db == 15.0
    assert noisy.utterance.snr_db == 15.0


def test_speaker_group_changes_acoustics_only():
    a = render(speaker_id=0, seed=15)
    b = render(speaker_id=1, seed=15)
    assert a.annotation == b.annotation
    assert a.utterance.segments[0].params != b.utterance.segments[0].params
