"""Frame-to-token alignment for synthetic utterances.

Segments are exact by construction, so every analysis frame maps to the
token whose segment contains the frame center; frames centered in a gap
map to :data:`NO_TOKEN`. This alignment turns token-level BIO annotations
into frame-level supervision for the slot head.
"""

from __future__ import annotations

import numpy as np

from slotvox.config.features import FeatureConfig
from slotvox.errors import ValidationError
from slotvox.features.frontend import log_mel_frames
from slotvox.synth.utterance import SyntheticUtterance
from slotvox.tagging.bio import format_tag, tags_to_spans

NO_TOKEN = -1


def frame_token_map(utterance: SyntheticUtterance, config: FeatureConfig) -> np.ndarray:
    """Per-frame token index (or :data:`NO_TOKEN`) under ``config``."""
    if not isinstance(utterance, SyntheticUtterance):
        raise ValidationError(
            f"utterance must be a SyntheticUtterance, got {type(utterance).__name__}"
        )
    if not isinstance(config, FeatureConfig):
        raise ValidationError(f"config must be a FeatureConfig, got {type(config).__name__}")
    if utterance.sample_rate != config.sample_rate:
        raise ValidationError(
            f"sample rate mismatch: utterance {utterance.sample_rate} "
            f"vs config {config.sample_rate}"
        )
    n_frames = log_mel_frames(utterance.n_samples, config)
    out = np.full(n_frames, NO_TOKEN, dtype=np.int64)
    centers = np.arange(n_frames, dtype=np.int64) * config.hop_length + config.frame_length // 2
    for index, segment in enumerate(utterance.segments):
        inside = (centers >= segment.start_sample) & (centers < segment.end_sample)
        out[inside] = index
    return out


def frame_tags(tags, token_map: np.ndarray) -> tuple[str, ...]:
    """Lift token-level BIO ``tags`` to per-frame tags (gaps become 'O')."""
    if isinstance(tags, (str, bytes)) or not isinstance(tags, (list, tuple)):
        raise ValidationError(f"tags must be a list/tuple, got {type(tags).__name__}")
    if not all(isinstance(tag, str) for tag in tags):
        raise ValidationError("tags must contain strings")
    if not isinstance(token_map, np.ndarray):
        raise ValidationError(f"token_map must be a numpy array, got {type(token_map).__name__}")
    if token_map.dtype.kind != "i":
        raise ValidationError(f"token_map must have an integer dtype, got {token_map.dtype}")
    if token_map.size and (int(token_map.min()) < NO_TOKEN or int(token_map.max()) >= len(tags)):
        raise ValidationError(f"token_map values must be within [{NO_TOKEN}, {len(tags) - 1}]")
    return tuple("O" if token == NO_TOKEN else tags[int(token)] for token in token_map)


def frame_span_tags(tags, token_map: np.ndarray) -> tuple[str, ...]:
    """Lift token-level BIO tags to canonical frame-level BIO spans.

    Each token span becomes exactly one frame span: ``B-<slot>`` on its
    first covered frame, ``I-<slot>`` on every following frame up to the
    last covered one — intra-span gap frames continue the span, so the
    result is always structurally valid BIO. Frames outside spans are
    ``O``; spans whose tokens no frame covers are dropped (a documented
    boundary for tokens shorter than one frame).
    """
    if isinstance(tags, (str, bytes)) or not isinstance(tags, (list, tuple)):
        raise ValidationError(f"tags must be a list/tuple, got {type(tags).__name__}")
    if not all(isinstance(tag, str) for tag in tags):
        raise ValidationError("tags must contain strings")
    if not isinstance(token_map, np.ndarray) or token_map.dtype.kind != "i":
        raise ValidationError("token_map must be an integer numpy array")
    if token_map.size and (int(token_map.min()) < NO_TOKEN or int(token_map.max()) >= len(tags)):
        raise ValidationError(f"token_map values must be within [{NO_TOKEN}, {len(tags) - 1}]")
    out = ["O"] * int(token_map.shape[0])
    for span in tags_to_spans(tuple(tags)):
        covered = np.flatnonzero((token_map >= span.start) & (token_map < span.end))
        if covered.size == 0:
            continue
        first, last = int(covered[0]), int(covered[-1])
        out[first] = format_tag("B", span.label)
        for frame in range(first + 1, last + 1):
            out[frame] = format_tag("I", span.label)
    return tuple(out)
