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

from slotvox.audio.pcm import float_to_pcm16, pcm16_to_float
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


def _parse_fmt(body: bytes) -> int:
    """Validate a 16-byte PCM fmt body; return the sample rate."""
    audio_format, channels, rate, byte_rate, block_align, bits = struct.unpack("<HHIIHH", body)
    if audio_format != 1:
        raise AudioError(f"only PCM format (1) is supported, got {audio_format}")
    if channels != 1:
        raise AudioError(f"only mono WAV is supported, got {channels} channels")
    if bits != 16:
        raise AudioError(f"only 16-bit WAV is supported, got {bits} bits")
    if not MIN_SAMPLE_RATE <= rate <= MAX_SAMPLE_RATE:
        raise AudioError(f"sample rate {rate} outside [{MIN_SAMPLE_RATE}, {MAX_SAMPLE_RATE}]")
    if byte_rate != rate * 2 or block_align != 2:
        raise AudioError("fmt byte_rate/block_align inconsistent with mono PCM16")
    return rate


def parse_wav(data: bytes) -> WavAudio:
    """Parse and strictly validate WAV bytes (canonical mono PCM16 only).

    Unknown chunks (LIST/INFO metadata, padding) are skipped by design;
    everything that affects the samples themselves must be exact.
    """
    if len(data) > MAX_WAV_BYTES:
        raise AudioError(f"WAV payload is {len(data)} bytes, exceeds cap {MAX_WAV_BYTES}")
    if len(data) < HEADER_SIZE:
        raise AudioError(f"WAV truncated: only {len(data)} bytes (need >= {HEADER_SIZE})")
    if data[:4] != b"RIFF":
        raise AudioError("not a RIFF file")
    if data[8:12] != b"WAVE":
        raise AudioError("not a RIFF/WAVE file")
    riff_size = struct.unpack("<I", data[4:8])[0]
    if riff_size + 8 != len(data):
        raise AudioError(f"RIFF size {riff_size} inconsistent with {len(data)} bytes")

    offset = 12
    sample_rate = -1
    pcm_data = b""
    fmt_seen = False
    data_seen = False
    while offset < len(data):
        if offset + 8 > len(data):
            raise AudioError("truncated chunk header")
        chunk_id = data[offset : offset + 4]
        chunk_size = struct.unpack("<I", data[offset + 4 : offset + 8])[0]
        body = offset + 8
        if body + chunk_size > len(data):
            raise AudioError(f"chunk {chunk_id!r} declares {chunk_size} bytes beyond file end")
        if chunk_id == b"fmt ":
            if fmt_seen:
                raise AudioError("duplicate fmt chunk")
            if chunk_size != 16:
                raise AudioError(f"only 16-byte PCM fmt chunks are supported, got {chunk_size}")
            sample_rate = _parse_fmt(data[body : body + 16])
            fmt_seen = True
        elif chunk_id == b"data":
            if data_seen:
                raise AudioError("duplicate data chunk")
            if not fmt_seen:
                raise AudioError("data chunk before fmt chunk")
            if chunk_size % 2:
                raise AudioError(f"data chunk size {chunk_size} is odd")
            pcm_data = data[body : body + chunk_size]
            data_seen = True
        offset = body + chunk_size + (chunk_size % 2)
    if not fmt_seen:
        raise AudioError("missing fmt chunk")
    if not data_seen or not pcm_data:
        raise AudioError("missing or empty data chunk")
    return WavAudio(samples=pcm16_to_float(pcm_data), sample_rate=sample_rate)


def read_wav(path: str | Path, *, max_bytes: int = MAX_WAV_BYTES) -> WavAudio:
    """Read a WAV file, enforcing ``max_bytes`` before loading the payload."""
    source = Path(path)
    if not source.is_file():
        raise AudioError(f"WAV file not found: {source}")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise AudioError(f"max_bytes must be an int, got {max_bytes!r}")
    if max_bytes <= 0 or max_bytes > MAX_WAV_BYTES:
        raise AudioError(f"max_bytes must be within (0, {MAX_WAV_BYTES}], got {max_bytes}")
    size = source.stat().st_size
    if size > max_bytes:
        raise AudioError(f"WAV file {source} is {size} bytes, exceeds bound {max_bytes}")
    return parse_wav(source.read_bytes())
