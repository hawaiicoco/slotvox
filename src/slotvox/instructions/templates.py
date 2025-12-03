"""Strict template rendering for instruction turns.

Templates carry ``{variable}`` placeholders (lowercase ids). Rendering
validates BOTH directions: every placeholder must be supplied and every
supplied variable must be used — no silent drops, no leftovers. Values
must be non-blank strings without braces, so a slot surface form can
never smuggle a placeholder into rendered text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from slotvox.data.factory import GeneratedExample
from slotvox.errors import ValidationError
from slotvox.instructions.schema import (
    TURN_ROLES,
    AudioRef,
    InstructionSample,
    Turn,
    check_provenance_hash,
)
from slotvox.schema.builtins import builtin_domain
from slotvox.schema.naming import validate_id

PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*)\}")


def template_variables(text: str) -> tuple[str, ...]:
    """Sorted unique ``{variable}`` names in ``text``; rejects stray braces."""
    if not isinstance(text, str) or not text.strip():
        raise ValidationError("template text must be a non-blank string")
    stripped = PLACEHOLDER.sub("", text)
    if "{" in stripped or "}" in stripped:
        raise ValidationError(f"template text has malformed braces: {text!r}")
    return tuple(sorted({match.group(1) for match in PLACEHOLDER.finditer(text)}))


@dataclass(frozen=True)
class TurnTemplate:
    """A role-tagged text template with ``{variable}`` placeholders."""

    template_id: str
    role: str
    text: str

    def __post_init__(self) -> None:
        validate_id(self.template_id, "TurnTemplate.template_id")
        if self.role not in TURN_ROLES:
            raise ValidationError(f"role must be one of {TURN_ROLES}, got {self.role!r}")
        template_variables(self.text)  # strict brace syntax

    @property
    def variables(self) -> tuple[str, ...]:
        """Sorted unique placeholder names."""
        return template_variables(self.text)

    def render(self, variables: Mapping[str, str]) -> Turn:
        """Fill every placeholder; strict in both directions."""
        if not isinstance(variables, Mapping):
            raise ValidationError(f"variables must be a mapping, got {type(variables).__name__}")
        expected = set(self.variables)
        supplied = set(variables)
        missing = sorted(expected - supplied)
        unknown = sorted(supplied - expected)
        if missing:
            raise ValidationError(f"template {self.template_id!r} is missing variables: {missing}")
        if unknown:
            raise ValidationError(f"template {self.template_id!r} got unknown variables: {unknown}")
        for name, value in variables.items():
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"variable {name!r} must be a non-blank string")
            if "{" in value or "}" in value:
                raise ValidationError(f"variable {name!r} must not contain braces")
        text = PLACEHOLDER.sub(lambda match: str(variables[match.group(1)]), self.text)
        return Turn(role=self.role, text=text)


SYSTEM_TEMPLATES = {
    "zh": TurnTemplate(
        "system-zh", "system", "你是语音助手。听用户的语音，识别{domain}领域的意图和槽位。"
    ),
    "en": TurnTemplate(
        "system-en",
        "system",
        "You are a speech assistant. Listen to the user's audio and identify "
        "the intent and slots for the {domain} domain.",
    ),
}
ASSISTANT_TEMPLATES = {
    "zh": TurnTemplate("answer-zh", "assistant", "意图：{intent}；槽位：{slots}"),
    "en": TurnTemplate("answer-en", "assistant", "Intent: {intent}; Slots: {slots}"),
}
EMPTY_SLOTS = {"zh": "无", "en": "none"}


def slots_text(slot_values: Mapping[str, tuple[str, ...]], slot_names: Sequence[str]) -> str:
    """Canonical slot rendering: domain order, ``slot=v1/v2``, joined ``", "``.

    Slots absent from ``slot_values`` are skipped; an empty mapping
    renders as ``""`` (callers substitute :data:`EMPTY_SLOTS`). Values
    must be non-blank and brace-free so the result can be re-rendered
    safely inside assistant templates.
    """
    if not isinstance(slot_values, Mapping):
        raise ValidationError(f"slot_values must be a mapping, got {type(slot_values).__name__}")
    if isinstance(slot_names, (str, bytes)) or not isinstance(slot_names, (list, tuple)):
        raise ValidationError("slot_names must be a list/tuple")
    names = tuple(slot_names)
    unknown = sorted(set(slot_values) - set(names))
    if unknown:
        raise ValidationError(f"slot_values reference unknown slots: {unknown}")
    parts = []
    for slot in names:
        if slot not in slot_values:
            continue
        values = slot_values[slot]
        if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)) or not values:
            raise ValidationError(f"slot {slot!r} values must be a non-empty list/tuple")
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"slot {slot!r} values must be non-blank strings")
            if "{" in value or "}" in value:
                raise ValidationError(f"slot {slot!r} value {value!r} must not contain braces")
        parts.append(f"{slot}={'/'.join(values)}")
    return ", ".join(parts)


def render_sample(
    example: GeneratedExample, *, domain: str, dataset_hash: str
) -> InstructionSample:
    """Render one generated example into an instruction sample.

    The user turn references the utterance audio by manifest id
    (``<domain>-<utterance_id>``); the assistant turn is the canonical
    structured answer built from the by-construction annotation. The
    transcript is deliberately NOT included — the task is speech to
    structure, and leaking text would make the audio channel optional.
    """
    if not isinstance(example, GeneratedExample):
        raise ValidationError(f"example must be a GeneratedExample, got {type(example).__name__}")
    spec = builtin_domain(domain)  # strict: unknown domains rejected
    check_provenance_hash(dataset_hash, "dataset_hash")
    language = example.annotation.language
    slots = slots_text(example.slot_values, spec.slot_names)
    if not slots:
        slots = EMPTY_SLOTS[language]
    manifest_id = f"{domain}-{example.utterance_id}"
    audio = AudioRef(
        manifest_id=manifest_id,
        dataset_hash=dataset_hash,
        duration_ms=example.utterance.n_samples / example.utterance.sample_rate * 1000.0,
    )
    turns = (
        SYSTEM_TEMPLATES[language].render({"domain": domain}),
        Turn(role="user", audio=audio),
        ASSISTANT_TEMPLATES[language].render({"intent": example.annotation.intent, "slots": slots}),
    )
    return InstructionSample(
        sample_id=manifest_id,
        domain=domain,
        language=language,
        intent=example.annotation.intent,
        turns=turns,
        tags=("synthetic",),
    )
