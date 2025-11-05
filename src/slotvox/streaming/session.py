"""Chunked streaming understanding with bounded buffers.

Semantics (all offline-comparable):

- :meth:`push_audio` appends to the bounded buffer and commits every
  frame whose audio AND lookahead have arrived; :meth:`finalize`
  commits the remaining complete frames (no lookahead at the end).
- Committed frames are encoded from a bounded context window (the TCN
  receptive field) or a carried hidden state (GRU); greedy slot tags
  are emitted per committed frame and never change afterwards.
- Latency accounting is in AUDIO time: milliseconds of pushed audio
  when the first frame committed.

Requires the torch extra and a model in eval mode. NOT real speech:
streams here are synthetic clips from the slotvox factory.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from slotvox.audio.framing import window as build_window
from slotvox.config import FeatureConfig, StreamConfig
from slotvox.errors import StreamingError
from slotvox.features.frontend import LOG_FLOOR
from slotvox.features.mel import mel_filterbank
from slotvox.features.spectrum import stft_power
from slotvox.models.joint import JointIntentSlotModel
from slotvox.streaming.buffer import AudioBuffer

STREAM_STATE_SCHEMA_ID = "slotvox.stream-state"
STREAM_STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StreamResult:
    """Final understanding of one streamed utterance."""

    frame_tags: tuple[str, ...]
    audio_ms: float
    first_frame_latency_ms: float | None

    @property
    def n_frames(self) -> int:
        """Number of committed frames."""
        return len(self.frame_tags)

    def to_dict(self) -> dict[str, Any]:
        """JSON-native dict form."""
        return {
            "frame_tags": list(self.frame_tags),
            "audio_ms": self.audio_ms,
            "first_frame_latency_ms": self.first_frame_latency_ms,
        }


class StreamSession:
    """Bounded streaming session for one utterance."""

    def __init__(
        self,
        model: JointIntentSlotModel,
        feature_config: FeatureConfig,
        stream_config: StreamConfig,
    ):
        if not isinstance(model, JointIntentSlotModel):
            raise StreamingError(
                f"model must be a JointIntentSlotModel, got {type(model).__name__}"
            )
        if not isinstance(feature_config, FeatureConfig):
            raise StreamingError("feature_config must be a FeatureConfig")
        if not isinstance(stream_config, StreamConfig):
            raise StreamingError("stream_config must be a StreamConfig")
        if feature_config.sample_rate != stream_config.sample_rate:
            raise StreamingError(
                f"sample rate mismatch: features {feature_config.sample_rate} "
                f"vs stream {stream_config.sample_rate}"
            )
        if model.training:
            raise StreamingError("streaming requires the model in eval mode")
        self._model = model
        self._features = feature_config
        self._stream = stream_config
        rate = feature_config.sample_rate
        self._rate = rate
        self._frame_length = feature_config.frame_length
        self._hop = feature_config.hop_length
        self._lookahead = stream_config.lookahead_ms * rate // 1000
        capacity_samples = stream_config.buffer_capacity_ms * rate // 1000
        self._buffer = AudioBuffer(capacity_samples)
        self._window = build_window(feature_config.window, self._frame_length)
        self._banks = mel_filterbank(
            rate,
            feature_config.n_fft,
            feature_config.n_mels,
            feature_config.fmin,
            feature_config.fmax,
        ).T
        receptive = getattr(model.encoder, "receptive_field", None)
        self._gru = receptive is None
        self._context = 0 if self._gru else int(receptive) - 1
        self._mel_window: deque[np.ndarray] = deque(maxlen=max(self._context, 1))
        self._hidden = None
        self._frame_cursor = 0
        self._n_pushed = 0
        self._committed_tags: list[str] = []
        self._first_frame_ms: float | None = None
        self._finished = False

    # -- introspection --------------------------------------------------

    @property
    def finished(self) -> bool:
        """True after :meth:`finalize`."""
        return self._finished

    @property
    def frames_committed(self) -> int:
        """Number of frames committed so far."""
        return self._frame_cursor

    @property
    def committed_tags(self) -> tuple[str, ...]:
        """Greedy slot tags emitted so far (never revised)."""
        return tuple(self._committed_tags)

    # -- streaming ------------------------------------------------------

    def push_audio(self, samples: np.ndarray) -> int:
        """Feed one audio chunk; returns the number of newly committed frames."""
        if self._finished:
            raise StreamingError("session is finished; create a new StreamSession")
        array = np.asarray(samples)
        self._buffer.append(array)
        self._n_pushed += int(array.size)
        return self._commit(final=False)

    def finalize(self) -> StreamResult:
        """Commit remaining complete frames and return the final result."""
        if self._finished:
            raise StreamingError("session already finalized")
        self._commit(final=True)
        self._finished = True
        if self._frame_cursor == 0:
            raise StreamingError(
                "no frames committed; the stream was shorter than one analysis frame"
            )
        return StreamResult(
            frame_tags=tuple(self._committed_tags),
            audio_ms=self._n_pushed / self._rate * 1000.0,
            first_frame_latency_ms=self._first_frame_ms,
        )

    # -- internals ------------------------------------------------------

    def _frames_available(self, final: bool) -> int:
        extra = 0 if final else self._lookahead
        if self._n_pushed < self._frame_cursor * self._hop + self._frame_length + extra:
            return 0
        ready = (self._n_pushed - extra - self._frame_length) // self._hop + 1
        return max(0, ready - self._frame_cursor)

    def _commit(self, final: bool) -> int:
        count = self._frames_available(final)
        if count == 0:
            return 0
        frames = np.empty((count, self._frame_length), dtype=np.float64)
        for index in range(count):
            frames[index] = self._buffer.peek()[
                index * self._hop : index * self._hop + self._frame_length
            ]
        self._buffer.consume(count * self._hop)
        mel = self._frame_mel(frames)
        states = self._encode(mel)
        slot_logits = self._model.slot_head(states)
        tags = [
            self._model.spec.tags[position] for position in slot_logits.argmax(dim=-1)[0].tolist()
        ]
        self._committed_tags.extend(tags)
        self._frame_cursor += count
        if self._first_frame_ms is None:
            self._first_frame_ms = self._n_pushed / self._rate * 1000.0
        return count

    def _frame_mel(self, frames: np.ndarray) -> torch.Tensor:
        windowed = frames * self._window
        power = stft_power(windowed, self._features.n_fft)
        mel = power @ self._banks
        logged = np.log(np.maximum(mel, LOG_FLOOR)).astype(np.float32)
        return torch.from_numpy(logged)

    def _encode(self, mel: torch.Tensor) -> torch.Tensor:
        """Encode new mel frames ``[n, F]`` to states shaped ``[1, n, H]``."""
        with torch.no_grad():
            if self._gru:
                states, self._hidden = self._model.encoder(mel.unsqueeze(0), self._hidden)
                return states
            context = list(self._mel_window)
            head = torch.from_numpy(np.stack(context)) if context else mel[:0]
            states = self._model.encoder(torch.cat([head, mel], dim=0).unsqueeze(0))
            for row in mel:
                self._mel_window.append(row.numpy().copy())
            return states[:, -mel.shape[0] :, :]
