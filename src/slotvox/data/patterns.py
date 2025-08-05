"""Scripted utterance templates: intent-labeled token patterns.

A template is a sequence of literal tokens and slot placeholders. Filling
placeholders from the lexicon produces the token sequence and BIO tags of
an :class:`AnnotatedUtterance`, so labels are correct BY CONSTRUCTION.
Templates are synthetic scripts — not sampled from any real dialogue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from slotvox.errors import ValidationError
from slotvox.schema.annotations import LANGUAGES
from slotvox.schema.builtins import builtin_domain
from slotvox.schema.domains import DomainSpec
from slotvox.schema.naming import validate_id


@dataclass(frozen=True)
class TokenPlan:
    """One template position: either literal text or a slot placeholder."""

    text: str | None = None
    slot: str | None = None

    def __post_init__(self) -> None:
        if (self.text is None) == (self.slot is None):
            raise ValidationError("TokenPlan needs exactly one of text/slot")
        if self.text is not None:
            if not isinstance(self.text, str) or not self.text:
                raise ValidationError("TokenPlan.text must be a non-empty string")
            if any(ch.isspace() for ch in self.text):
                raise ValidationError(f"literal token {self.text!r} must not contain whitespace")
        else:
            validate_id(self.slot, "TokenPlan.slot")

    @property
    def is_slot(self) -> bool:
        """True for slot placeholders, False for literal tokens."""
        return self.text is None


def _lit(*texts: str) -> tuple[TokenPlan, ...]:
    """Shorthand for a run of literal tokens."""
    return tuple(TokenPlan(text=text) for text in texts)


def _slot(name: str) -> TokenPlan:
    """Shorthand for a slot placeholder."""
    return TokenPlan(slot=name)


@dataclass(frozen=True)
class PatternTemplate:
    """A scripted token pattern bound to one intent."""

    pattern_id: str
    intent: str
    tokens: tuple[TokenPlan, ...]
    language: str = "zh"

    def __post_init__(self) -> None:
        if not isinstance(self.pattern_id, str) or not self.pattern_id:
            raise ValidationError("PatternTemplate.pattern_id must be a non-empty string")
        if any(ch.isspace() for ch in self.pattern_id):
            raise ValidationError("PatternTemplate.pattern_id must not contain whitespace")
        parts = self.pattern_id.split("/")
        if len(parts) != 3 or not all(parts) or not parts[2].isdigit():
            raise ValidationError(
                f"pattern_id must be '<domain>/<intent>/<index>', got {self.pattern_id!r}"
            )
        validate_id(self.intent, "PatternTemplate.intent")
        if parts[1] != self.intent:
            raise ValidationError(
                f"pattern_id intent {parts[1]!r} does not match intent {self.intent!r}"
            )
        if self.language not in LANGUAGES:
            raise ValidationError(f"language must be one of {LANGUAGES}, got {self.language!r}")
        tokens = self.tokens
        if isinstance(tokens, list):
            tokens = tuple(tokens)
            object.__setattr__(self, "tokens", tokens)
        if not isinstance(tokens, tuple) or not tokens:
            raise ValidationError("PatternTemplate.tokens must be a non-empty tuple/list")
        previous_slot: str | None = None
        for token in tokens:
            if not isinstance(token, TokenPlan):
                raise ValidationError("PatternTemplate.tokens must contain TokenPlan instances")
            if token.is_slot and token.slot == previous_slot:
                raise ValidationError(
                    f"adjacent placeholders for slot {token.slot!r} would merge spans"
                )
            previous_slot = token.slot if token.is_slot else None

    def validate_against(self, domain: DomainSpec) -> None:
        """Check the template against a domain, strictly in both directions.

        The pattern_id must name this domain, the intent must exist, every
        placeholder must reference a slot the intent declares, and every
        required slot of the intent must appear in the template.
        """
        if not isinstance(domain, DomainSpec):
            raise ValidationError(
                f"validate_against expects a DomainSpec, got {type(domain).__name__}"
            )
        domain_part = self.pattern_id.split("/")[0]
        if domain_part != domain.domain:
            raise ValidationError(f"pattern_id names domain {domain_part!r}, got {domain.domain!r}")
        try:
            intent = domain.intent(self.intent)
        except ValidationError as exc:
            raise ValidationError(
                f"template {self.pattern_id!r} has unknown intent {self.intent!r}"
            ) from exc
        declared = set(intent.slot_names)
        seen: set[str] = set()
        for token in self.tokens:
            if token.is_slot:
                if token.slot not in declared:
                    raise ValidationError(
                        f"template {self.pattern_id!r} references slot {token.slot!r} "
                        f"not declared by intent {self.intent!r}"
                    )
                seen.add(token.slot)
        missing = [name for name in intent.required_slots if name not in seen]
        if missing:
            raise ValidationError(f"template {self.pattern_id!r} omits required slots: {missing}")


_TEMPLATES: dict[tuple[str, str], tuple[PatternTemplate, ...]] = {}


def register_templates(domain_id: str, language: str, templates: Any) -> None:
    """Validate and register the template table for one domain/language."""
    domain = builtin_domain(domain_id)
    if language not in LANGUAGES:
        raise ValidationError(f"language must be one of {LANGUAGES}, got {language!r}")
    key = (domain_id, language)
    if key in _TEMPLATES:
        raise ValidationError(f"templates already registered for {key}")
    if isinstance(templates, (str, bytes)) or not isinstance(templates, (list, tuple)):
        raise ValidationError("templates must be a list/tuple of PatternTemplate")
    items = tuple(templates)
    if not items:
        raise ValidationError(f"need at least one template for {key}")
    seen: set[str] = set()
    for template in items:
        if not isinstance(template, PatternTemplate):
            raise ValidationError("templates must contain PatternTemplate instances")
        template.validate_against(domain)
        if template.pattern_id in seen:
            raise ValidationError(f"duplicate pattern_id {template.pattern_id!r}")
        seen.add(template.pattern_id)
    _TEMPLATES[key] = items


def templates_for(domain_id: str, language: str) -> tuple[PatternTemplate, ...]:
    """Strict lookup of the template table for one domain/language."""
    items = _TEMPLATES.get((domain_id, language))
    if items is None:
        raise ValidationError(
            f"no templates registered for domain {domain_id!r} language {language!r}"
        )
    return items


def registered_templates() -> tuple[tuple[str, str], ...]:
    """All (domain, language) keys with registered templates, sorted."""
    return tuple(sorted(_TEMPLATES))
