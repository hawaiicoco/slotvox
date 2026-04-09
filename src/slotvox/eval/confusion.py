"""Label confusion matrices with golden ordering and per-label scores.

Rows are gold labels, columns are predicted labels, and the label order
is fixed at construction (default: sorted union of observed labels) so
matrices are diffable across runs. Unknown labels are rejected in BOTH
directions — gold and predicted — against explicit label sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from slotvox.errors import ValidationError


def _check_label(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{name} must be a non-empty string, got {value!r}")
    return value


@dataclass(frozen=True)
class ConfusionMatrix:
    """Rows = gold labels, columns = predicted labels."""

    labels: tuple[str, ...]
    counts: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        labels = self.labels
        if isinstance(labels, list):
            labels = tuple(labels)
            object.__setattr__(self, "labels", labels)
        if isinstance(labels, (str, bytes)) or not isinstance(labels, tuple):
            raise ValidationError("labels must be a tuple/list of strings")
        for label in labels:
            _check_label(label, "label")
        if len(set(labels)) != len(labels):
            raise ValidationError("labels must be unique")
        counts = self.counts
        if isinstance(counts, list):
            counts = tuple(tuple(row) if isinstance(row, list) else row for row in counts)
            object.__setattr__(self, "counts", counts)
        if not isinstance(counts, tuple) or len(counts) != len(labels):
            raise ValidationError(
                f"counts must be a square tuple matching labels ({len(labels)} rows)"
            )
        for row in counts:
            if not isinstance(row, tuple) or len(row) != len(labels):
                raise ValidationError(f"counts rows must have {len(labels)} columns")
            for value in row:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValidationError(f"counts must be non-negative ints, got {value!r}")

    @classmethod
    def from_pairs(cls, gold, pred, labels=None) -> ConfusionMatrix:
        """Build from aligned label sequences; ``labels`` pins the order."""
        for name, sequence in (("gold", gold), ("pred", pred)):
            if isinstance(sequence, (str, bytes)) or not isinstance(sequence, (list, tuple)):
                raise ValidationError(f"{name} must be a list/tuple, got {type(sequence).__name__}")
        if len(gold) != len(pred):
            raise ValidationError(f"gold and pred must align ({len(gold)} vs {len(pred)})")
        for name, sequence in (("gold", gold), ("pred", pred)):
            for value in sequence:
                _check_label(value, f"{name} label")
        if labels is None:
            resolved = tuple(sorted(set(gold) | set(pred)))
        else:
            if isinstance(labels, (str, bytes)) or not isinstance(labels, (list, tuple)):
                raise ValidationError("labels must be a tuple/list of strings")
            resolved = tuple(labels)
            for label in resolved:
                _check_label(label, "label")
            if len(set(resolved)) != len(resolved):
                raise ValidationError("labels must be unique")
            index = set(resolved)
            for name, sequence in (("gold", gold), ("pred", pred)):
                unknown = sorted({value for value in sequence if value not in index})
                if unknown:
                    raise ValidationError(f"{name} pairs reference unknown labels: {unknown}")
        if not resolved:
            raise ValidationError("confusion matrix needs at least one pair or explicit labels")
        positions = {label: index for index, label in enumerate(resolved)}
        grid = [[0] * len(resolved) for _ in resolved]
        for gold_label, pred_label in zip(gold, pred, strict=True):
            grid[positions[gold_label]][positions[pred_label]] += 1
        return cls(labels=resolved, counts=tuple(tuple(row) for row in grid))

    @property
    def total(self) -> int:
        """Total pair count."""
        return sum(sum(row) for row in self.counts)

    @property
    def accuracy(self) -> float:
        """Trace / total (0.0 for an empty matrix — documented boundary)."""
        if self.total == 0:
            return 0.0
        trace = sum(self.counts[index][index] for index in range(len(self.labels)))
        return trace / self.total

    def per_label(self) -> dict[str, dict[str, Any]]:
        """Per-label precision/recall/f1/support in ``labels`` order.

        Zero denominators give 0.0 (documented): a label never predicted
        has precision 0, a label never in gold has recall 0.
        """
        stats: dict[str, dict[str, Any]] = {}
        for index, label in enumerate(self.labels):
            true_positive = self.counts[index][index]
            predicted = sum(row[index] for row in self.counts)
            support = sum(self.counts[index])
            precision = true_positive / predicted if predicted else 0.0
            recall = true_positive / support if support else 0.0
            total = precision + recall
            f1 = 2.0 * precision * recall / total if total else 0.0
            stats[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        return stats

    def to_dict(self) -> dict[str, Any]:
        """JSON-native form."""
        return {"labels": list(self.labels), "counts": [list(row) for row in self.counts]}

    @classmethod
    def from_dict(cls, data: Any) -> ConfusionMatrix:
        """Reconstruct from :meth:`to_dict` output (strict)."""
        if not isinstance(data, dict) or set(data) != {"labels", "counts"}:
            raise ValidationError(
                "confusion payload must be a dict with exactly 'labels' and 'counts'"
            )
        if not isinstance(data["labels"], list) or not isinstance(data["counts"], list):
            raise ValidationError("confusion 'labels' and 'counts' must be lists")
        return cls(labels=tuple(data["labels"]), counts=tuple(tuple(r) for r in data["counts"]))
