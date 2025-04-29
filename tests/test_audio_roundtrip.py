"""read_wav/write_wav file roundtrip and bound enforcement."""

import numpy as np
import pytest

from slotvox.audio.riff import MAX_WAV_BYTES, read_wav, write_wav
from slotvox.audio.wav import WavAudio
from slotvox.errors import AudioError


def test_file_round_trip_within_quantization(tmp_path):
    samples = np.sin(np.linspace(0, 8 * np.pi, 1600)).astype(np.float32)
    audio = WavAudio(samples, 16000)
    path = write_wav(tmp_path / "tone.wav", audio)
    recovered = read_wav(path)
    assert recovered.sample_rate == 16000
    assert np.allclose(recovered.samples, samples, atol=1e-4)


def test_lossless_for_exactly_representable_samples(tmp_path):
    audio = WavAudio(np.zeros(8, dtype=np.float32), 16000)
    assert read_wav(write_wav(tmp_path / "zeros.wav", audio)) == audio


def test_write_creates_parent_directories(tmp_path):
    audio = WavAudio(np.zeros(4, dtype=np.float32), 16000)
    path = write_wav(tmp_path / "nested" / "dir" / "a.wav", audio)
    assert path.is_file()


def test_size_bound_enforced_before_parsing(tmp_path):
    path = tmp_path / "big.wav"
    path.write_bytes(b"RIFF" + b"\x00" * 100)
    with pytest.raises(AudioError, match="exceeds bound"):
        read_wav(path, max_bytes=64)


def test_invalid_max_bytes_rejected(tmp_path):
    path = write_wav(tmp_path / "ok.wav", WavAudio(np.zeros(4, dtype=np.float32), 16000))
    with pytest.raises(AudioError, match="max_bytes"):
        read_wav(path, max_bytes=0)
    with pytest.raises(AudioError, match="max_bytes"):
        read_wav(path, max_bytes=MAX_WAV_BYTES + 1)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(AudioError, match="not found"):
        read_wav(tmp_path / "absent.wav")
