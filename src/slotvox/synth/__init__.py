"""Deterministic synthetic signal and utterance builders (NOT real speech)."""

from slotvox.synth.mix import apply_snr, fade_edges
from slotvox.synth.segments import SIGNAL_KINDS, Segment, render_segment, validate_params
from slotvox.synth.signals import (
    MAX_FORMANTS,
    chirp,
    formant_pattern,
    noise,
    silence,
    time_base,
    tone,
)
from slotvox.synth.utterance import SegmentPlan, SyntheticUtterance, render_utterance

__all__ = [
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
