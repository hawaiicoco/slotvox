"""WAV writer byte-structure contracts, verified by parsing the header."""

import struct

import numpy as np
import pytest

from slotvox.audio.pcm import float_to_pcm16
from slotvox.audio.riff import build_wav_bytes
from slotvox.errors import AudioError


def test_header_fields_describe_mono_pcm16():
    pcm = float_to_pcm16(np.zeros(10, dtype=np.float32))
    blob = build_wav_bytes(pcm, 16000)
    assert blob[:4] == b"RIFF"
    assert blob[8:12] == b"WAVE"
    assert blob[12:16] == b"fmt "
    riff_size = struct.unpack("<I", blob[4:8])[0]
    assert riff_size + 8 == len(blob)
    assert struct.unpack("<I", blob[16:20])[0] == 16
    assert struct.unpack("<HHIIHH", blob[20:36]) == (1, 1, 16000, 32000, 2, 16)
    assert blob[36:40] == b"data"
    assert struct.unpack("<I", blob[40:44])[0] == len(pcm)
    assert blob[44:] == pcm


def test_writer_rejects_invalid_inputs():
    with pytest.raises(AudioError, match="zero frames"):
        build_wav_bytes(b"", 16000)
    with pytest.raises(AudioError, match="even length"):
        build_wav_bytes(b"\x00", 16000)
    with pytest.raises(AudioError, match="sample_rate"):
        build_wav_bytes(b"\x00\x01", 999)
    with pytest.raises(AudioError, match="sample_rate"):
        build_wav_bytes(b"\x00\x01", 16000.0)
