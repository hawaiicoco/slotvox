"""Golden pin for the audio package API surface."""

import numpy as np

from slotvox import audio

GOLDEN_ALL = [
    "MAX_WAV_BYTES",
    "WINDOW_NAMES",
    "WavAudio",
    "build_wav_bytes",
    "float_to_pcm16",
    "frame_count",
    "frame_signal",
    "parse_wav",
    "pcm16_to_float",
    "read_wav",
    "window",
    "write_wav",
]


def test_audio_api_surface_is_pinned():
    assert sorted(audio.__all__) == GOLDEN_ALL
    for name in GOLDEN_ALL:
        assert hasattr(audio, name)


def test_cross_module_round_trip_still_holds(tmp_path):
    samples = np.linspace(-0.5, 0.5, 64, dtype=np.float32)
    clip = audio.WavAudio(samples, 16000)
    recovered = audio.read_wav(audio.write_wav(tmp_path / "x.wav", clip))
    assert np.allclose(recovered.samples, samples, atol=1e-4)
    frames = audio.frame_signal(recovered.samples, 16, 8)
    windowed = frames * audio.window("hann", 16)
    assert windowed.shape == frames.shape
