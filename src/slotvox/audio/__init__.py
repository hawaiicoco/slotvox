"""Audio primitives: strict bounded WAV I/O, framing, and windows."""

from slotvox.audio.framing import WINDOW_NAMES, frame_count, frame_signal, window
from slotvox.audio.pcm import float_to_pcm16, pcm16_to_float
from slotvox.audio.riff import MAX_WAV_BYTES, build_wav_bytes, parse_wav, read_wav, write_wav
from slotvox.audio.wav import WavAudio

__all__ = [
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
