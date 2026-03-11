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

from dataclasses import dataclass

from slotvox.errors import ValidationError
from slotvox.tagging.bio import Span

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


def span_f1(gold_spans, pred_spans) -> SlotScore:
    """Strict span matching: label and both boundaries must be equal.

    Matching is multiplicity-aware: two identical gold spans need two
    identical predicted spans for full credit.
    """
    gold = _check_spans(gold_spans, "gold_spans")
    pred = _check_spans(pred_spans, "pred_spans")
    remaining = list(pred)
    credit = 0.0
    for span in gold:
        for index, candidate in enumerate(remaining):
            if candidate == span:
                credit += 1.0
                remaining.pop(index)
                break
    return SlotScore(gold_count=len(gold), pred_count=len(pred), credit=credit)
