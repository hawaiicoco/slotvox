"""Golden pin for the synth package API surface and honesty wording."""

import slotvox.synth as synth
import slotvox.synth.signals as signals
import slotvox.synth.utterance as utterance

GOLDEN_ALL = [
    "MAX_FORMANTS",
    "SIGNAL_KINDS",
    "Segment",
    "SegmentPlan",
    "SyntheticUtterance",
    "apply_snr",
    "chirp",
    "fade_edges",
    "formant_pattern",
    "noise",
    "render_segment",
    "render_utterance",
    "silence",
    "time_base",
    "tone",
    "validate_params",
]


def test_synth_api_surface_is_pinned():
    assert sorted(synth.__all__) == GOLDEN_ALL
    for name in GOLDEN_ALL:
        assert hasattr(synth, name)


def test_modules_document_synthetic_nature():
    for module in (synth, signals, utterance):
        doc = (module.__doc__ or "").lower()
        assert "real speech" in doc, f"{module.__name__} must state it is not real speech"
