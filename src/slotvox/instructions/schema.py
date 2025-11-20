"""Interleaved speech-text instruction samples for speech-LLM training.

The schema follows the message-turn convention of instruction-tuned chat
models (``system`` / ``user`` / ``assistant``) and references audio by
manifest id — samples never embed waveforms or features. Consumers
resolve ids against a slotvox dataset or feature-store manifest. Every
structure is JSON-native, strictly validated in both directions, and
schema-versioned so serialized corpora pin their format.

This module produces DATA for documented training protocols; slotvox
ships no model weights and downloads nothing. Task shape follows the
joint intent-slot literature (arXiv:1902.10909); layout conventions are
informed by instruction-tuning data formats (arXiv:2104.06678) and the
HuggingFace Datasets docs — the implementation is original.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from slotvox.errors import ValidationError
from slotvox.schema.annotations import LANGUAGES
from slotvox.schema.naming import validate_id
from slotvox.util.jsoncanon import stable_hash

INSTRUCTION_SCHEMA_ID = "slotvox.instruction-sample"
INSTRUCTION_SCHEMA_VERSION = 1
TURN_ROLES = ("system", "user", "assistant")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_DURATION_MS = 600_000.0


def check_provenance_hash(value: Any, field: str) -> str:
    """Validate a 64-char lowercase hex provenance hash."""
    if not isinstance(value, str) or not _HASH_PATTERN.fullmatch(value):
        raise ValidationError(f"{field} must be a 64-char lowercase hex hash, got {value!r}")
    return value


def _exact_keys(data: Any, required: set[str], label: str) -> None:
    """Require ``data`` to be a dict with exactly the ``required`` keys."""
    if not isinstance(data, dict):
        raise ValidationError(f"{label} payload must be a dict, got {type(data).__name__}")
    unknown = sorted(set(data) - required)
    missing = sorted(required - set(data))
    if unknown:
        raise ValidationError(f"{label} got unknown keys: {unknown}")
    if missing:
        raise ValidationError(f"{label} is missing keys: {missing}")


@dataclass(frozen=True)
class AudioRef:
    """A reference to one manifest utterance (never inline samples).

    ``dataset_hash`` pins the provenance: it is the ``dataset_hash`` of
    the generated dataset the utterance came from, so a corpus cannot
    silently mix references from different datasets.
    """

    manifest_id: str
    dataset_hash: str
    duration_ms: float

    def __post_init__(self) -> None:
        validate_id(self.manifest_id, "AudioRef.manifest_id")
        check_provenance_hash(self.dataset_hash, "AudioRef.dataset_hash")
        if isinstance(self.duration_ms, bool) or not isinstance(self.duration_ms, (int, float)):
            raise ValidationError(f"duration_ms must be a number, got {self.duration_ms!r}")
        duration = float(self.duration_ms)
        if not 0.0 < duration <= MAX_DURATION_MS:
            raise ValidationError(
                f"duration_ms must be within (0, {MAX_DURATION_MS:g}], got {self.duration_ms!r}"
            )
        object.__setattr__(self, "duration_ms", duration)

    def to_dict(self) -> dict[str, Any]:
        """JSON-native dict form."""
        return {
            "manifest_id": self.manifest_id,
            "dataset_hash": self.dataset_hash,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: Any) -> AudioRef:
        """Reconstruct from :meth:`to_dict` output (strict)."""
        _exact_keys(data, {"manifest_id", "dataset_hash", "duration_ms"}, "audio ref")
        return cls(
            manifest_id=data["manifest_id"],
            dataset_hash=data["dataset_hash"],
            duration_ms=data["duration_ms"],
        )


@dataclass(frozen=True)
class Turn:
    """One conversation turn: text, audio (user turns only), or both.

    Audio is restricted to user turns by construction: the task shape is
    speech in (user) and structured text out (assistant), with an
    optional text-only system prompt.
    """

    role: str
    text: str | None = None
    audio: AudioRef | None = None

    def __post_init__(self) -> None:
        if self.role not in TURN_ROLES:
            raise ValidationError(f"role must be one of {TURN_ROLES}, got {self.role!r}")
        if self.text is not None and (not isinstance(self.text, str) or not self.text.strip()):
            raise ValidationError("turn text must be a non-blank string or None")
        if self.audio is not None:
            if not isinstance(self.audio, AudioRef):
                raise ValidationError(
                    f"audio must be an AudioRef or None, got {type(self.audio).__name__}"
                )
            if self.role != "user":
                raise ValidationError("only user turns may carry audio")
        if self.text is None and self.audio is None:
            raise ValidationError("a turn needs text, audio, or both")

    def to_dict(self) -> dict[str, Any]:
        """JSON-native dict form."""
        return {
            "role": self.role,
            "text": self.text,
            "audio": None if self.audio is None else self.audio.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Any) -> Turn:
        """Reconstruct from :meth:`to_dict` output (strict)."""
        _exact_keys(data, {"role", "text", "audio"}, "turn")
        audio = data["audio"]
        return cls(
            role=data["role"],
            text=data["text"],
            audio=None if audio is None else AudioRef.from_dict(audio),
        )


@dataclass(frozen=True)
class InstructionSample:
    """A complete interleaved speech-text instruction sample.

    Structure rules (all enforced): an optional single leading ``system``
    turn, then strict ``user``/``assistant`` alternation ending on the
    assistant, and at least one user turn carrying audio — a sample
    without speech is not a speech-instruction sample.
    """

    sample_id: str
    domain: str
    language: str
    intent: str
    turns: tuple[Turn, ...]
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_id(self.sample_id, "InstructionSample.sample_id")
        validate_id(self.domain, "InstructionSample.domain")
        validate_id(self.intent, "InstructionSample.intent")
        if self.language not in LANGUAGES:
            raise ValidationError(f"language must be one of {LANGUAGES}, got {self.language!r}")
        turns = self.turns
        if isinstance(turns, list):
            turns = tuple(turns)
            object.__setattr__(self, "turns", turns)
        if not isinstance(turns, tuple) or not all(isinstance(turn, Turn) for turn in turns):
            raise ValidationError("turns must be a tuple/list of Turn")
        roles = [turn.role for turn in turns]
        body = roles[1:] if roles and roles[0] == "system" else roles
        if roles and roles[0] == "system" and "system" in body:
            raise ValidationError("at most one system turn is allowed, and it must be first")
        if len(body) < 2 or len(body) % 2 != 0:
            raise ValidationError(
                "turns must alternate user/assistant in pairs after the optional system turn"
            )
        for position, role in enumerate(body):
            expected = "user" if position % 2 == 0 else "assistant"
            if role != expected:
                raise ValidationError(
                    f"turn {position} of the body must be {expected!r}, got {role!r}"
                )
        if not any(turn.audio is not None for turn in turns):
            raise ValidationError("an instruction sample needs at least one audio turn")
        tags = self.tags
        if isinstance(tags, list):
            tags = tuple(tags)
            object.__setattr__(self, "tags", tags)
        if not isinstance(tags, tuple):
            raise ValidationError("tags must be a tuple/list of ids")
        for tag in tags:
            validate_id(tag, "InstructionSample tag")
        if len(set(tags)) != len(tags):
            raise ValidationError("tags must be unique")

    def to_dict(self) -> dict[str, Any]:
        """JSON-native dict under the instruction-sample envelope."""
        return {
            "schema": INSTRUCTION_SCHEMA_ID,
            "schema_version": INSTRUCTION_SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "domain": self.domain,
            "language": self.language,
            "intent": self.intent,
            "turns": [turn.to_dict() for turn in self.turns],
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Any) -> InstructionSample:
        """Reconstruct from :meth:`to_dict` output (strict)."""
        _exact_keys(
            data,
            {
                "schema",
                "schema_version",
                "sample_id",
                "domain",
                "language",
                "intent",
                "turns",
                "tags",
            },
            "instruction sample",
        )
        if data["schema"] != INSTRUCTION_SCHEMA_ID:
            raise ValidationError(f"unknown instruction schema {data['schema']!r}")
        if data["schema_version"] != INSTRUCTION_SCHEMA_VERSION:
            raise ValidationError(
                f"unsupported instruction schema version {data['schema_version']!r}"
            )
        if not isinstance(data["turns"], list):
            raise ValidationError("instruction sample 'turns' must be a list")
        if not isinstance(data["tags"], list):
            raise ValidationError("instruction sample 'tags' must be a list")
        return cls(
            sample_id=data["sample_id"],
            domain=data["domain"],
            language=data["language"],
            intent=data["intent"],
            turns=tuple(Turn.from_dict(turn) for turn in data["turns"]),
            tags=tuple(data["tags"]),
        )

    @property
    def sample_hash(self) -> str:
        """Provenance hash of the full serialized sample."""
        return stable_hash(self.to_dict())
