"""Core evaluation metrics for joint intent-slot understanding.

Slot scoring has two documented boundary policies — ``strict`` (exact
label + boundaries) and ``partial`` (proportional boundary-overlap
credit) — and both are reported side by side wherever slots are scored;
they are never averaged into one number. All metrics are pure functions
over validated inputs; every golden value in the tests is hand-computed.

Empty-side conventions (documented boundaries): a side with zero spans
scores 1.0 for its ratio (nothing missed / nothing spurious), so two
empty sides agree perfectly with F1 1.0.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from slotvox.errors import ValidationError
from slotvox.schema.naming import validate_id
from slotvox.tagging.bio import Span, tags_to_spans

SLOT_POLICIES = ("strict", "partial")


@dataclass(frozen=True)
class SlotScore:
    """Span-matching totals: summed ``credit`` over gold/pred counts.

    ``credit`` is 0/1 per matched span under ``strict`` and a Jaccard
    overlap fraction under ``partial``.
    """

    gold_count: int
    pred_count: int
    credit: float

    def __post_init__(self) -> None:
        for name in ("gold_count", "pred_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValidationError(f"{name} must be a non-negative int, got {value!r}")
        if isinstance(self.credit, bool) or not isinstance(self.credit, (int, float)):
            raise ValidationError(f"credit must be a number, got {self.credit!r}")
        credit = float(self.credit)
        bound = float(min(self.gold_count, self.pred_count))
        if credit < 0.0 or credit > bound + 1e-9:
            raise ValidationError(f"credit must be within [0, {bound}], got {self.credit!r}")
        object.__setattr__(self, "credit", credit)

    @property
    def precision(self) -> float:
        """credit / pred_count (1.0 when nothing was predicted)."""
        return self.credit / self.pred_count if self.pred_count else 1.0

    @property
    def recall(self) -> float:
        """credit / gold_count (1.0 when nothing was gold)."""
        return self.credit / self.gold_count if self.gold_count else 1.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall (0.0 when both are 0)."""
        total = self.precision + self.recall
        if total == 0.0:
            return 0.0
        return 2.0 * self.precision * self.recall / total


def _check_spans(spans, name: str) -> tuple[Span, ...]:
    if isinstance(spans, (str, bytes)) or not isinstance(spans, (list, tuple)):
        raise ValidationError(f"{name} must be a list/tuple, got {type(spans).__name__}")
    checked: list[Span] = []
    for item in spans:
        if isinstance(item, Span):
            checked.append(item)
        elif isinstance(item, tuple) and len(item) == 3:
            checked.append(Span(*item))
        else:
            raise ValidationError(f"{name} must contain Span or (label, start, end), got {item!r}")
    return tuple(checked)


def check_policy(policy: str) -> str:
    """Validate a slot boundary policy name."""
    if policy not in SLOT_POLICIES:
        raise ValidationError(f"policy must be one of {SLOT_POLICIES}, got {policy!r}")
    return policy


def span_f1(gold_spans, pred_spans, policy: str = "strict") -> SlotScore:
    """Score predicted spans against gold under a documented policy.

    ``strict`` requires the label and both boundaries to match exactly
    (multiplicity-aware). ``partial`` awards Jaccard credit
    (overlap / union) to same-label spans that overlap, assigned greedily
    best-credit-first with positional tie-breaks, so results are
    deterministic.
    """
    check_policy(policy)
    gold = _check_spans(gold_spans, "gold_spans")
    pred = _check_spans(pred_spans, "pred_spans")
    if policy == "strict":
        remaining = list(pred)
        credit = 0.0
        for span in gold:
            for index, candidate in enumerate(remaining):
                if candidate == span:
                    credit += 1.0
                    remaining.pop(index)
                    break
        return SlotScore(gold_count=len(gold), pred_count=len(pred), credit=credit)
    pairs = []
    for gold_index, gold_span in enumerate(gold):
        for pred_index, pred_span in enumerate(pred):
            if gold_span.label != pred_span.label:
                continue
            overlap = min(gold_span.end, pred_span.end) - max(gold_span.start, pred_span.start)
            if overlap <= 0:
                continue
            union = max(gold_span.end, pred_span.end) - min(gold_span.start, pred_span.start)
            pairs.append((-overlap / union, gold_index, pred_index))
    pairs.sort()
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    credit = 0.0
    for negated, gold_index, pred_index in pairs:
        if gold_index in used_gold or pred_index in used_pred:
            continue
        used_gold.add(gold_index)
        used_pred.add(pred_index)
        credit += -negated
    return SlotScore(gold_count=len(gold), pred_count=len(pred), credit=credit)


@dataclass(frozen=True)
class IntentScore:
    """Intent-classification totals."""

    correct: int
    total: int

    def __post_init__(self) -> None:
        for name in ("correct", "total"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValidationError(f"{name} must be a non-negative int, got {value!r}")
        if self.correct > self.total:
            raise ValidationError(f"correct {self.correct} exceeds total {self.total}")

    @property
    def accuracy(self) -> float:
        """correct / total (0.0 for an empty total)."""
        return self.correct / self.total if self.total else 0.0


def _check_labels(sequence, name: str) -> tuple[str, ...]:
    if isinstance(sequence, (str, bytes)) or not isinstance(sequence, (list, tuple)):
        raise ValidationError(f"{name} must be a list/tuple, got {type(sequence).__name__}")
    for item in sequence:
        if not isinstance(item, str) or not item:
            raise ValidationError(f"{name} must contain non-empty strings, got {item!r}")
    return tuple(sequence)


def intent_accuracy(gold, pred) -> IntentScore:
    """Intent accuracy over aligned gold/pred label sequences."""
    gold_labels = _check_labels(gold, "gold")
    pred_labels = _check_labels(pred, "pred")
    if len(gold_labels) != len(pred_labels):
        raise ValidationError(
            f"gold and pred must align ({len(gold_labels)} vs {len(pred_labels)})"
        )
    if not gold_labels:
        raise ValidationError("intent_accuracy needs at least one pair")
    correct = sum(1 for left, right in zip(gold_labels, pred_labels, strict=True) if left == right)
    return IntentScore(correct=correct, total=len(gold_labels))


def slot_f1(gold_tags, pred_tags, policy: str = "strict") -> SlotScore:
    """Sequence-level slot scoring: BIO tag sequences -> spans -> :func:`span_f1`.

    Works for token-level and frame-level tags alike (both are BIO
    sequences). Structurally invalid sequences raise ``TaggingError``;
    repair them first if the evaluation policy tolerates raw model output.
    """
    check_policy(policy)
    gold_seq = _check_labels(gold_tags, "gold_tags")
    pred_seq = _check_labels(pred_tags, "pred_tags")
    if len(gold_seq) != len(pred_seq):
        raise ValidationError(
            f"gold and pred tag sequences must align ({len(gold_seq)} vs {len(pred_seq)})"
        )
    return span_f1(tags_to_spans(gold_seq), tags_to_spans(pred_seq), policy)


def slice_pairs(mapping) -> tuple[tuple[str, str], ...]:
    """Canonicalize a slice mapping into sorted ``(kind, value)`` pairs."""
    if not isinstance(mapping, Mapping):
        raise ValidationError(f"slices must be a mapping, got {type(mapping).__name__}")
    pairs = []
    for kind, value in mapping.items():
        validate_id(kind, "slice kind")
        if not isinstance(value, str) or not value:
            raise ValidationError(f"slice value for {kind!r} must be a non-empty string")
        pairs.append((kind, value))
    return tuple(sorted(pairs))


@dataclass(frozen=True)
class TurnRecord:
    """One evaluated turn: intents, tag sequences, and slice metadata."""

    gold_intent: str
    pred_intent: str
    gold_tags: tuple[str, ...]
    pred_tags: tuple[str, ...]
    slices: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in ("gold_intent", "pred_intent"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or any(ch.isspace() for ch in value):
                raise ValidationError(f"{name} must be a non-empty string without whitespace")
        for name in ("gold_tags", "pred_tags"):
            tags = getattr(self, name)
            if isinstance(tags, list):
                tags = tuple(tags)
                object.__setattr__(self, name, tags)
            _check_labels(tags, name)
        slices = self.slices
        if isinstance(slices, Mapping):
            slices = slice_pairs(slices)
        if isinstance(slices, list):
            slices = tuple(slices)
        if not isinstance(slices, tuple):
            raise ValidationError("slices must be a mapping or tuple/list of pairs")
        kinds = []
        normalized = []
        for pair in slices:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValidationError(f"slice entries must be (kind, value) pairs, got {pair!r}")
            kind, value = pair
            validate_id(kind, "slice kind")
            if not isinstance(value, str) or not value:
                raise ValidationError(f"slice value for {kind!r} must be a non-empty string")
            if kind in kinds:
                raise ValidationError(f"duplicate slice kind {kind!r}")
            kinds.append(kind)
            normalized.append((kind, value))
        object.__setattr__(self, "slices", tuple(sorted(normalized)))

    @property
    def slice_mapping(self) -> dict[str, str]:
        """Slices as a plain dict."""
        return dict(self.slices)


@dataclass(frozen=True)
class TurnScore:
    """Joint turn totals (intent AND slots strictly right)."""

    correct: int
    total: int

    def __post_init__(self) -> None:
        for name in ("correct", "total"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValidationError(f"{name} must be a non-negative int, got {value!r}")
        if self.correct > self.total:
            raise ValidationError(f"correct {self.correct} exceeds total {self.total}")

    @property
    def accuracy(self) -> float:
        """correct / total (0.0 for an empty total)."""
        return self.correct / self.total if self.total else 0.0


def _require_records(records) -> list[TurnRecord]:
    if isinstance(records, (str, bytes)) or not isinstance(records, (list, tuple)) or not records:
        raise ValidationError("records must be a non-empty list/tuple of TurnRecord")
    for record in records:
        if not isinstance(record, TurnRecord):
            raise ValidationError(f"records must contain TurnRecord, got {type(record).__name__}")
    return list(records)


def turn_accuracy(records) -> TurnScore:
    """Joint accuracy: a turn counts only when the intent is right AND the
    strict span sets are equal (the documented turn-level policy)."""
    items = _require_records(records)
    correct = 0
    for record in items:
        if record.gold_intent != record.pred_intent:
            continue
        if tags_to_spans(record.gold_tags) == tags_to_spans(record.pred_tags):
            correct += 1
    return TurnScore(correct=correct, total=len(items))


def micro_slot_f1(records, policy: str = "strict") -> SlotScore:
    """Pool span credit over all records (micro averaging)."""
    check_policy(policy)
    items = _require_records(records)
    gold_total = 0
    pred_total = 0
    credit_total = 0.0
    for record in items:
        score = slot_f1(record.gold_tags, record.pred_tags, policy)
        gold_total += score.gold_count
        pred_total += score.pred_count
        credit_total += score.credit
    return SlotScore(gold_count=gold_total, pred_count=pred_total, credit=credit_total)
