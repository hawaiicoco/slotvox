"""Chunked streaming understanding with bounded buffers.

Semantics (all offline-comparable):

- :meth:`push_audio` appends to the bounded buffer and commits every
  frame whose audio AND lookahead have arrived; :meth:`finalize`
  commits the remaining complete frames (no lookahead at the end).
- Committed frames are encoded from a bounded context window (the TCN
  receptive field) or a carried hidden state (GRU); greedy slot tags
  are emitted per committed frame and never change afterwards.
- Intent hypotheses are partial: recomputed from accumulated frame
  states, flagged stable once the softmax posterior reaches
  ``stability_threshold``; a later different stable intent counts as a
  revision (reported, never hidden).
- Latency accounting is in AUDIO time: milliseconds pushed when the
  first frame committed and when the first stable intent appeared.

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
class IntentHypothesis:
    """The current partial intent estimate."""

    intent: str
    posterior: float
    stable: bool
    frames_seen: int


@dataclass(frozen=True)
class StreamResult:
    """Final understanding of one streamed utterance."""

    intent: str
    posterior: float
    stable: bool
    frame_tags: tuple[str, ...]
    audio_ms: float
    first_frame_latency_ms: float | None
    first_stable_latency_ms: float | None
    revisions: int

    @property
    def n_frames(self) -> int:
        """Number of committed frames."""
        return len(self.frame_tags)

    def to_dict(self) -> dict[str, Any]:
        """JSON-native dict form."""
        return {
            "intent": self.intent,
            "posterior": self.posterior,
            "stable": self.stable,
            "frame_tags": list(self.frame_tags),
            "audio_ms": self.audio_ms,
            "first_frame_latency_ms": self.first_frame_latency_ms,
            "first_stable_latency_ms": self.first_stable_latency_ms,
            "revisions": self.revisions,
        }


class StreamSession:
    """Bounded, resumable streaming session for one utterance."""

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
        self._state_sum = torch.zeros(model.spec.train_config.hidden_size)
        self._state_count = 0
        self._hypothesis: IntentHypothesis | None = None
        self._stable_intent: str | None = None
        self._revisions = 0
        self._first_frame_ms: float | None = None
        self._first_stable_ms: float | None = None
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
    def hypothesis(self) -> IntentHypothesis | None:
        """Current partial intent hypothesis (None before the first frame)."""
        return self._hypothesis

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
        if self._state_count == 0:
            raise StreamingError(
                "no frames committed; the stream was shorter than one analysis frame"
            )
        hypothesis = self._hypothesis
        assert hypothesis is not None  # guaranteed by state_count > 0
        return StreamResult(
            intent=hypothesis.intent,
            posterior=hypothesis.posterior,
            stable=hypothesis.stable,
            frame_tags=tuple(self._committed_tags),
            audio_ms=self._n_pushed / self._rate * 1000.0,
            first_frame_latency_ms=self._first_frame_ms,
            first_stable_latency_ms=self._first_stable_ms,
            revisions=self._revisions,
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
        self._accumulate_intent(states)
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

    def _accumulate_intent(self, states: torch.Tensor) -> None:
        with torch.no_grad():
            self._state_sum = self._state_sum + states[0].sum(dim=0)
            self._state_count += int(states.shape[1])
            logits = self._model.intent_head.from_accumulated(
                self._state_sum.unsqueeze(0),
                torch.tensor([self._state_count]),
            )[0]
            posterior = torch.softmax(logits, dim=-1)
            best = int(posterior.argmax())
            confidence = float(posterior[best])
            stable = confidence >= self._stream.stability_threshold
            intent = self._model.spec.intents[best]
            if stable:
                if self._stable_intent is None:
                    self._first_stable_ms = self._n_pushed / self._rate * 1000.0
                elif self._stable_intent != intent:
                    self._revisions += 1
                self._stable_intent = intent
            self._hypothesis = IntentHypothesis(
                intent=intent,
                posterior=confidence,
                stable=stable,
                frames_seen=self._state_count,
            )

    # -- resumable state --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Versioned resumable snapshot (canonical-JSON-ready)."""
        return {
            "schema": STREAM_STATE_SCHEMA_ID,
            "schema_version": STREAM_STATE_SCHEMA_VERSION,
            "model_hash": self._model.spec.model_hash,
            "feature_config": self._features.to_dict(),
            "stream_config": self._stream.to_dict(),
            "buffer": self._buffer.peek().tolist(),
            "n_pushed": self._n_pushed,
            "frame_cursor": self._frame_cursor,
            "committed_tags": list(self._committed_tags),
            "mel_window": [row.tolist() for row in self._mel_window],
            "state_sum": self._state_sum.tolist(),
            "state_count": self._state_count,
            "hidden": None if self._hidden is None else self._hidden.tolist(),
            "stable_intent": self._stable_intent,
            "revisions": self._revisions,
            "first_frame_ms": self._first_frame_ms,
            "first_stable_ms": self._first_stable_ms,
            "finished": self._finished,
        }

    @classmethod
    def from_dict(cls, data: Any, model: JointIntentSlotModel) -> StreamSession:
        """Rebuild a session from :meth:`to_dict` output (strict).

        The ``model_hash`` guard rejects states produced by a different
        model; counters, buffer, and window contents are validated before
        anything is restored. The partial hypothesis is recomputed from
        the restored accumulator, so a resumed session continues exactly
        where the original stopped.
        """
        if not isinstance(data, dict):
            raise StreamingError(f"stream state must be a dict, got {type(data).__name__}")
        required = {
            "schema",
            "schema_version",
            "model_hash",
            "feature_config",
            "stream_config",
            "buffer",
            "n_pushed",
            "frame_cursor",
            "committed_tags",
            "mel_window",
            "state_sum",
            "state_count",
            "hidden",
            "stable_intent",
            "revisions",
            "first_frame_ms",
            "first_stable_ms",
            "finished",
        }
        unknown = sorted(set(data) - required)
        missing = sorted(required - set(data))
        if unknown:
            raise StreamingError(f"stream state got unknown keys: {unknown}")
        if missing:
            raise StreamingError(f"stream state is missing keys: {missing}")
        if data["schema"] != STREAM_STATE_SCHEMA_ID:
            raise StreamingError(f"unknown stream state schema {data['schema']!r}")
        if data["schema_version"] != STREAM_STATE_SCHEMA_VERSION:
            raise StreamingError(f"unsupported stream state version {data['schema_version']!r}")
        if not isinstance(model, JointIntentSlotModel):
            raise StreamingError(
                f"model must be a JointIntentSlotModel, got {type(model).__name__}"
            )
        if data["model_hash"] != model.spec.model_hash:
            raise StreamingError("stream state was produced by a different model")
        features = FeatureConfig.from_dict(data["feature_config"])
        stream = StreamConfig.from_dict(data["stream_config"])
        session = cls(model, features, stream)
        frame_cursor = data["frame_cursor"]
        n_pushed = data["n_pushed"]
        state_count = data["state_count"]
        committed = data["committed_tags"]
        for name, value in (
            ("frame_cursor", frame_cursor),
            ("n_pushed", n_pushed),
            ("state_count", state_count),
            ("revisions", data["revisions"]),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise StreamingError(f"stream state {name} must be a non-negative int")
        if not isinstance(committed, list) or not all(isinstance(tag, str) for tag in committed):
            raise StreamingError("stream state committed_tags must be a list of strings")
        if len(committed) != frame_cursor or state_count != frame_cursor:
            raise StreamingError("stream state counters disagree with committed tags")
        if n_pushed < frame_cursor * session._hop:
            raise StreamingError("stream state n_pushed is behind the frame cursor")
        buffer = np.asarray(data["buffer"], dtype=np.float32)
        if buffer.ndim != 1 or buffer.size > session._buffer.capacity:
            raise StreamingError("stream state buffer is malformed or over capacity")
        if buffer.size:
            session._buffer.append(buffer)
        rows = data["mel_window"]
        if not isinstance(rows, list) or not all(
            isinstance(row, list) and len(row) == features.n_mels for row in rows
        ):
            raise StreamingError(
                f"stream state mel_window must be a list of {features.n_mels}-bin rows"
            )
        session._mel_window.extend(np.asarray(row, dtype=np.float32) for row in rows)
        hidden = data["hidden"]
        if hidden is not None:
            try:
                tensor = torch.tensor(hidden, dtype=torch.float32)
            except (RuntimeError, TypeError, ValueError) as exc:
                raise StreamingError("stream state hidden is malformed") from exc
            if not bool(torch.isfinite(tensor).all()):
                raise StreamingError("stream state hidden contains non-finite values")
            session._hidden = tensor
        state_sum = np.asarray(data["state_sum"], dtype=np.float64)
        hidden_size = model.spec.train_config.hidden_size
        if state_sum.shape != (hidden_size,) or not bool(np.isfinite(state_sum).all()):
            raise StreamingError(
                "stream state state_sum must be a finite vector of hidden_size values"
            )
        session._frame_cursor = frame_cursor
        session._n_pushed = n_pushed
        session._committed_tags = list(committed)
        session._state_sum = torch.tensor(state_sum, dtype=torch.float32)
        session._state_count = state_count
        stable_intent = data["stable_intent"]
        if stable_intent is not None and stable_intent not in model.spec.intents:
            raise StreamingError(f"unknown stable intent {stable_intent!r} in stream state")
        session._stable_intent = stable_intent
        session._revisions = data["revisions"]
        for name in ("first_frame_ms", "first_stable_ms"):
            value = data[name]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
            ):
                raise StreamingError(f"stream state {name} must be None or a non-negative number")
        session._first_frame_ms = (
            None if data["first_frame_ms"] is None else float(data["first_frame_ms"])
        )
        session._first_stable_ms = (
            None if data["first_stable_ms"] is None else float(data["first_stable_ms"])
        )
        if not isinstance(data["finished"], bool):
            raise StreamingError("stream state finished must be a bool")
        session._finished = data["finished"]
        if session._state_count:
            with torch.no_grad():
                logits = model.intent_head.from_accumulated(
                    session._state_sum.unsqueeze(0),
                    torch.tensor([session._state_count]),
                )[0]
                posterior = torch.softmax(logits, dim=-1)
                best = int(posterior.argmax())
                confidence = float(posterior[best])
                session._hypothesis = IntentHypothesis(
                    intent=model.spec.intents[best],
                    posterior=confidence,
                    stable=confidence >= stream.stability_threshold,
                    frames_seen=state_count,
                )
        return session
