"""Deterministic token -> acoustic segment mapping.

Each token of a synthetic utterance becomes one signal segment whose
parameters derive from stable hashes: identical tokens always sound
identical, and filler tokens (low-band tones) are acoustically distinct
from slot-value tokens (two-formant stacks). Slot identity shifts the
formant band, value identity shifts the details. "Speaker groups" apply a
deterministic frequency multiplier — simulated acoustic variation between
groups, not real speakers. None of this resembles real speech.
"""

from __future__ import annotations

from slotvox.errors import ValidationError
from slotvox.schema.naming import validate_id
from slotvox.synth.utterance import SegmentPlan
from slotvox.util.jsoncanon import stable_hash

FILLER_BAND = (150.0, 400.0)
SLOT_BASE = 400.0
SLOT_STEP = 120.0
SLOT_BANDS = 8
FORMANT_JITTER = 60.0
MAX_SPEAKER_SHIFT = 0.12


def speaker_shift(speaker_id: int) -> float:
    """Deterministic frequency multiplier within 1 ± MAX_SPEAKER_SHIFT."""
    if isinstance(speaker_id, bool) or not isinstance(speaker_id, int) or speaker_id < 0:
        raise ValidationError(f"speaker_id must be a non-negative int, got {speaker_id!r}")
    frac = int(stable_hash(["speaker-shift", speaker_id])[:8], 16) / 0xFFFFFFFF
    return round(1.0 + MAX_SPEAKER_SHIFT * (2.0 * frac - 1.0), 6)


def _digest_fraction(digest: str, chars: int, span: float) -> float:
    return int(digest[:chars], 16) % 10**6 / 10**6 * span


def token_segment_plan(
    domain_id: str,
    token_text: str,
    slot: str | None,
    *,
    duration_ms: int = 80,
    speaker_id: int = 0,
) -> SegmentPlan:
    """Build the acoustic plan for one token (label↔sound binding)."""
    if not isinstance(domain_id, str) or not domain_id:
        raise ValidationError("domain_id must be a non-empty string")
    if not isinstance(token_text, str) or not token_text:
        raise ValidationError("token_text must be a non-empty string")
    if any(ch.isspace() for ch in token_text):
        raise ValidationError(f"token_text must not contain whitespace, got {token_text!r}")
    if (
        isinstance(duration_ms, bool)
        or not isinstance(duration_ms, int)
        or not 10 <= duration_ms <= 2000
    ):
        raise ValidationError(f"duration_ms must be within [10, 2000], got {duration_ms!r}")
    shift = speaker_shift(speaker_id)
    if slot is None:
        digest = stable_hash(["filler", domain_id, token_text])
        freq = FILLER_BAND[0] + _digest_fraction(digest, 8, FILLER_BAND[1] - FILLER_BAND[0])
        return SegmentPlan(
            label=f"lit:{token_text}",
            kind="tone",
            duration_ms=duration_ms,
            params={"freq": round(freq * shift, 2)},
        )
    validate_id(slot, "token_segment_plan slot")
    slot_digest = stable_hash(["slot-band", slot])
    value_digest = stable_hash(["value", domain_id, slot, token_text])
    band = int(slot_digest[:8], 16) % SLOT_BANDS
    first = SLOT_BASE + band * SLOT_STEP + int(value_digest[:4], 16) % int(FORMANT_JITTER)
    second = first * (2.2 + int(value_digest[4:8], 16) % 100 / 100.0 * 0.6)
    formants = [round(first * shift, 2), round(second * shift, 2)]
    return SegmentPlan(
        label=f"{slot}:{token_text}",
        kind="formant",
        duration_ms=duration_ms,
        params={"formants": formants},
    )
