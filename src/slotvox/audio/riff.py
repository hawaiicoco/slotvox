"""Strict, bounded RIFF/WAVE (mono PCM16) writing and reading.

Only the minimal profile slotvox generates and consumes is supported. The
writer emits canonical 44-byte headers; the reader validates structure and
bounds before allocating sample memory and rejects anything ambiguous
instead of guessing. Inputs are capped at ``MAX_WAV_BYTES`` so hostile
files cannot exhaust memory.
"""

from __future__ import annotations

import struct
from pathlib import Path

from slotvox.audio.pcm import float_to_pcm16
from slotvox.audio.wav import MAX_SAMPLE_RATE, MIN_SAMPLE_RATE, WavAudio
from slotvox.errors import AudioError

MAX_WAV_BYTES = 64 * 1024 * 1024  # hard cap on WAV inputs (bytes)
HEADER_SIZE = 44


def build_wav_bytes(pcm_data: bytes, sample_rate: int) -> bytes:
    """Assemble RIFF/WAVE bytes for mono PCM16 ``pcm_data``."""
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise AudioError(f"sample_rate must be an int, got {sample_rate!r}")
    if not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE:
        raise AudioError(
            f"sample_rate must be within [{MIN_SAMPLE_RATE}, {MAX_SAMPLE_RATE}], got {sample_rate}"
        )
    if not pcm_data:
        raise AudioError("cannot write a WAV with zero frames")
    if len(pcm_data) % 2:
        raise AudioError(f"PCM16 data must have even length, got {len(pcm_data)}")
    fmt = struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16)
    riff_size = 4 + 8 + len(fmt) + 8 + len(pcm_data)
    return (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(pcm_data))
        + pcm_data
    )


def write_wav(path: str | Path, audio: WavAudio) -> Path:
    """Write ``audio`` as a mono PCM16 WAV file and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(build_wav_bytes(float_to_pcm16(audio.samples), audio.sample_rate))
    return target
