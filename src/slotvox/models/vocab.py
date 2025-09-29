"""Vocabulary construction for intents and BIO frame tags.

Vocabs derive deterministically from a DomainSpec: intents in declaration
order; tags as ``("O", B-<slot>, I-<slot>, ...)`` following the domain's
slot declaration order. :data:`PAD_INDEX` is the cross-entropy ignore
label used for padded frames and is never part of a vocab.
"""

from __future__ import annotations

from slotvox.errors import ValidationError
from slotvox.schema.domains import DomainSpec
from slotvox.tagging.bio import format_tag

PAD_INDEX = -100


def build_intent_vocab(domain: DomainSpec) -> tuple[str, ...]:
    """Intent names in the domain's declaration order."""
    _require_domain(domain)
    return domain.intent_names


def build_tag_vocab(domain: DomainSpec) -> tuple[str, ...]:
    """BIO tag names: 'O' first, then B-/I- per slot in declaration order."""
    _require_domain(domain)
    tags = ["O"]
    for slot in domain.slot_names:
        tags.append(format_tag("B", slot))
        tags.append(format_tag("I", slot))
    return tuple(tags)


def _require_domain(domain: DomainSpec) -> None:
    if not isinstance(domain, DomainSpec):
        raise ValidationError(f"domain must be a DomainSpec, got {type(domain).__name__}")


class Vocab:
    """A strict bidirectional name <-> index mapping."""

    __slots__ = ("_names", "_index")

    def __init__(self, names):
        if isinstance(names, (str, bytes)) or not isinstance(names, (list, tuple)) or not names:
            raise ValidationError("vocab needs a non-empty list/tuple of names")
        for name in names:
            if not isinstance(name, str) or not name:
                raise ValidationError(f"vocab names must be non-empty strings, got {name!r}")
        if len(set(names)) != len(names):
            raise ValidationError("vocab names must be unique")
        self._names = tuple(names)
        self._index = {name: index for index, name in enumerate(self._names)}

    @property
    def names(self) -> tuple[str, ...]:
        """All names in index order."""
        return self._names

    def __len__(self) -> int:
        return len(self._names)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vocab):
            return NotImplemented
        return self._names == other._names

    def __repr__(self) -> str:
        return f"Vocab({list(self._names)!r})"

    def index(self, name: str) -> int:
        """Name -> index (strict)."""
        try:
            return self._index[name]
        except (KeyError, TypeError) as exc:
            raise ValidationError(f"unknown vocab entry {name!r}") from exc

    def name(self, index: int) -> str:
        """Index -> name (strict)."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValidationError(f"vocab index must be an int, got {index!r}")
        if not 0 <= index < len(self._names):
            raise ValidationError(f"vocab index {index} out of range [0, {len(self._names)})")
        return self._names[index]

    def indices(self, names) -> tuple[int, ...]:
        """Map a sequence of names to indices (strict)."""
        if isinstance(names, (str, bytes)) or not isinstance(names, (list, tuple)):
            raise ValidationError(f"names must be a list/tuple, got {type(names).__name__}")
        return tuple(self.index(name) for name in names)

    def to_list(self) -> list[str]:
        """JSON-native form."""
        return list(self._names)

    @classmethod
    def from_list(cls, names) -> Vocab:
        """Rebuild from :meth:`to_list` output (validates again)."""
        if not isinstance(names, list):
            raise ValidationError(f"vocab payload must be a list, got {type(names).__name__}")
        return cls(names)
