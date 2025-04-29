"""Strict WAV parser: acceptance and rejection contracts for raw bytes."""

import struct

import numpy as np
import pytest

from slotvox.audio.pcm import float_to_pcm16
from slotvox.audio.riff import build_wav_bytes, parse_wav
from slotvox.errors import AudioError

FMT = struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
DATA = float_to_pcm16(np.zeros(4, dtype=np.float32))


def riff(chunks):
    body = b"WAVE" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def chunk(chunk_id, payload):
    return chunk_id + struct.pack("<I", len(payload)) + payload


def good(n=8, rate=16000):
    return build_wav_bytes(float_to_pcm16(np.zeros(n, dtype=np.float32)), rate)


def corrupt(blob, offset, replacement):
    data = bytearray(blob)
    data[offset : offset + len(replacement)] = replacement
    return bytes(data)


def test_accepts_canonical_file():
    audio = parse_wav(good())
    assert audio.sample_rate == 16000
    assert audio.n_samples == 8


def test_rejects_non_riff_and_non_wave():
    with pytest.raises(AudioError, match="not a RIFF file"):
        parse_wav(b"XIFF" + good()[4:])
    with pytest.raises(AudioError, match="RIFF/WAVE"):
        parse_wav(corrupt(good(), 8, b"AVI "))


def test_rejects_truncation_and_inconsistent_riff_size():
    with pytest.raises(AudioError, match="truncated"):
        parse_wav(good()[:30])
    with pytest.raises(AudioError, match="inconsistent"):
        parse_wav(corrupt(good(), 4, struct.pack("<I", 10)))


def test_rejects_unsupported_profiles():
    blob = good()
    with pytest.raises(AudioError, match="PCM"):
        parse_wav(corrupt(blob, 20, struct.pack("<H", 3)))
    with pytest.raises(AudioError, match="mono"):
        parse_wav(corrupt(blob, 22, struct.pack("<H", 2)))
    with pytest.raises(AudioError, match="16-bit"):
        parse_wav(corrupt(blob, 34, struct.pack("<H", 8)))
    with pytest.raises(AudioError, match="sample rate"):
        parse_wav(corrupt(blob, 24, struct.pack("<I", 500)))
    with pytest.raises(AudioError, match="byte_rate"):
        parse_wav(corrupt(blob, 28, struct.pack("<I", 31999)))


def test_rejects_missing_or_misordered_chunks():
    big = float_to_pcm16(np.zeros(32, dtype=np.float32))  # keeps files >= 44 bytes
    with pytest.raises(AudioError, match="before fmt"):
        parse_wav(riff([chunk(b"data", big)]))
    with pytest.raises(AudioError, match="before fmt"):
        parse_wav(riff([chunk(b"data", big), chunk(b"fmt ", FMT)]))
    with pytest.raises(AudioError, match="duplicate fmt"):
        parse_wav(riff([chunk(b"fmt ", FMT), chunk(b"fmt ", FMT), chunk(b"data", big)]))
    with pytest.raises(AudioError, match="missing fmt"):
        parse_wav(riff([chunk(b"LIST", b"INFO" * 8)]))
    with pytest.raises(AudioError, match="missing or empty data"):
        parse_wav(riff([chunk(b"fmt ", FMT), chunk(b"LIST", b"INFO")]))
    with pytest.raises(AudioError, match="duplicate data"):
        parse_wav(riff([chunk(b"fmt ", FMT), chunk(b"data", big), chunk(b"data", big)]))


def test_rejects_truncated_chunk_header():
    body = b"WAVE" + chunk(b"fmt ", FMT) + chunk(b"LIST", b"INFO" * 8) + b"dat"
    blob = b"RIFF" + struct.pack("<I", len(body)) + body
    with pytest.raises(AudioError, match="truncated chunk header"):
        parse_wav(blob)


def test_rejects_chunk_overflow_and_odd_data():
    oversized = riff([chunk(b"fmt ", FMT), b"data" + struct.pack("<I", len(DATA) + 100) + DATA])
    with pytest.raises(AudioError, match="beyond file end"):
        parse_wav(oversized)
    odd = riff([chunk(b"fmt ", FMT), b"data" + struct.pack("<I", 3) + b"\x01\x02\x03"])
    with pytest.raises(AudioError, match="odd"):
        parse_wav(odd)


def test_skips_unknown_chunks():
    blob = riff([chunk(b"LIST", b"INFO"), chunk(b"fmt ", FMT), chunk(b"data", DATA)])
    assert parse_wav(blob).n_samples == 4
