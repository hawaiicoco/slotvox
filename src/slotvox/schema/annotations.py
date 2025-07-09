"""Utterance↔annotation pairs: tokens + intent + per-token BIO tags.

An annotation binds a tokenized transcription to an intent and one BIO tag
per token. Tag *structure* (span validity, repairs) lives in
:mod:`slotvox.tagging`; this module enforces shape and vocabulary.
Transcriptions here are synthetic experiment fixtures — never transcripts
of real speech.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from slotvox.errors import ValidationError
from slotvox.schema.naming import validate_id

LANGUAGES = ("zh", "en")
MAX_TOKENS = 1000
MAX_ID_CHARS = 128
_TAG_PATTERN = re.compile(r"^(B|I)-(.+)$")


@dataclass(frozen=True)
class AnnotatedUtterance:
    """One utterance↔annotation pair."""

    utterance_id: str
    language: str
    tokens: tuple[str, ...]
    tags: tuple[str, ...]
    intent: str

    def __post_init__(self) -> None:
        if not isinstance(self.utterance_id, str) or not self.utterance_id:
            raise ValidationError("AnnotatedUtterance.utterance_id must be a non-empty string")
        if len(self.utterance_id) > MAX_ID_CHARS:
            raise ValidationError(
                f"AnnotatedUtterance.utterance_id must be <= {MAX_ID_CHARS} characters"
            )
        if any(ch.isspace() for ch in self.utterance_id):
            raise ValidationError("AnnotatedUtterance.utterance_id must not contain whitespace")
        if self.language not in LANGUAGES:
            raise ValidationError(
                f"AnnotatedUtterance.language must be one of {LANGUAGES}, got {self.language!r}"
            )
        validate_id(self.intent, "AnnotatedUtterance.intent")
        for field_name in ("tokens", "tags"):
            value = getattr(self, field_name)
            if isinstance(value, list):
                object.__setattr__(self, field_name, tuple(value))
            if not isinstance(getattr(self, field_name), tuple):
                raise ValidationError(f"AnnotatedUtterance.{field_name} must be a tuple/list")
        if not 1 <= len(self.tokens) <= MAX_TOKENS:
            raise ValidationError(
                f"token count must be within [1, {MAX_TOKENS}], got {len(self.tokens)}"
            )
        for token in self.tokens:
            if not isinstance(token, str) or not token:
                raise ValidationError("tokens must be non-empty strings")
            if any(ch.isspace() for ch in token):
                raise ValidationError(f"token {token!r} must not contain whitespace")
        if len(self.tags) != len(self.tokens):
            raise ValidationError(
                f"tags must align with tokens ({len(self.tags)} vs {len(self.tokens)})"
            )
        for tag in self.tags:
            if tag == "O":
                continue
            if not isinstance(tag, str):
                raise ValidationError(f"tags must be strings, got {type(tag).__name__}")
            match = _TAG_PATTERN.fullmatch(tag)
            if match is None:
                raise ValidationError(f"tag {tag!r} must be 'O' or start with 'B-'/'I-'")
            validate_id(match.group(2), f"tag slot name in {tag!r}")

    @property
    def text(self) -> str:
        """Transcription text; zh joins without spaces, en with spaces."""
        return "".join(self.tokens) if self.language == "zh" else " ".join(self.tokens)

    def tag_slot_names(self) -> tuple[str, ...]:
        """Sorted unique slot names appearing in tags."""
        return tuple(sorted({tag[2:] for tag in self.tags if tag != "O"}))
