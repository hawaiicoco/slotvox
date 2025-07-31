"""Language-aware tokenization for transcription text.

Chinese text is tokenized per character; English text per whitespace-
delimited word. Input is NFC-normalized first, so equivalent unicode
sequences tokenize identically. Whitespace never becomes a token. This is
a deterministic tokenizer for synthetic fixtures — not a general-purpose
segmenter, and it makes no linguistic claims beyond the two rules above.
"""

from __future__ import annotations

import unicodedata

from slotvox.errors import ValidationError
from slotvox.schema.annotations import LANGUAGES


def is_cjk(char: str) -> bool:
    """True for CJK ideographs, CJK punctuation, and fullwidth forms."""
    if not isinstance(char, str) or len(char) != 1:
        raise ValidationError(f"is_cjk expects a single character, got {char!r}")
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3000 <= code <= 0x303F
        or 0xFF00 <= code <= 0xFFEF
    )


def tokenize(text: str, language: str) -> tuple[str, ...]:
    """Tokenize ``text`` for ``language`` ("zh" per char, "en" per word)."""
    if not isinstance(text, str):
        raise ValidationError(f"text must be a string, got {type(text).__name__}")
    if language not in LANGUAGES:
        raise ValidationError(f"language must be one of {LANGUAGES}, got {language!r}")
    normalized = unicodedata.normalize("NFC", text)
    if not normalized.strip():
        raise ValidationError("tokenize requires non-empty text (whitespace does not count)")
    if language == "zh":
        return tuple(char for char in normalized if not char.isspace())
    return tuple(normalized.split())
