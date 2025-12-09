"""Per-sample quality flags for instruction corpora.

Flags are honest about missing data: without an SNR sidecar the SNR flag
stays ``None`` (unknown), never silently ``True``. Thresholds are
explicit and validated, and ``issues`` carries stable machine-readable
ids (``duration-out-of-bounds``, ``snr-below-minimum``,
``answer-not-canonical``) so exports can summarize without parsing prose.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from slotvox.errors import ValidationError
from slotvox.instructions.schema import MAX_DURATION_MS, InstructionSample
from slotvox.instructions.templates import ASSISTANT_TEMPLATES

ISSUE_DURATION = "duration-out-of-bounds"
ISSUE_SNR = "snr-below-minimum"
ISSUE_ANSWER = "answer-not-canonical"


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"{name} must be a finite number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class QualityThresholds:
    """Explicit bounds for corpus quality flags."""

    min_duration_ms: float = 200.0
    max_duration_ms: float = 30000.0
    min_snr_db: float | None = None

    def __post_init__(self) -> None:
        low = _finite_number("min_duration_ms", self.min_duration_ms)
        high = _finite_number("max_duration_ms", self.max_duration_ms)
        object.__setattr__(self, "min_duration_ms", low)
        object.__setattr__(self, "max_duration_ms", high)
        if not 0.0 < low < high <= MAX_DURATION_MS:
            raise ValidationError(
                f"durations must satisfy 0 < min < max <= {MAX_DURATION_MS:g}, got {low} / {high}"
            )
        if self.min_snr_db is not None:
            object.__setattr__(self, "min_snr_db", _finite_number("min_snr_db", self.min_snr_db))


@dataclass(frozen=True)
class QualityFlags:
    """Per-sample verdict; ``issues`` empty means the sample passed."""

    sample_id: str
    duration_ok: bool
    snr_ok: bool | None
    answer_canonical: bool
    issues: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValidationError("sample_id must be a non-empty string")
        for name in ("duration_ok", "answer_canonical"):
            if not isinstance(getattr(self, name), bool):
                raise ValidationError(f"{name} must be a bool")
        if self.snr_ok is not None and not isinstance(self.snr_ok, bool):
            raise ValidationError("snr_ok must be a bool or None")
        issues = self.issues
        if isinstance(issues, list):
            issues = tuple(issues)
            object.__setattr__(self, "issues", issues)
        if not isinstance(issues, tuple) or not all(
            isinstance(issue, str) and issue for issue in issues
        ):
            raise ValidationError("issues must be a tuple/list of non-empty strings")

    @property
    def ok(self) -> bool:
        """No issues at all."""
        return not self.issues


@dataclass(frozen=True)
class QualityReport:
    """Aggregate verdicts over a corpus."""

    flags: tuple[QualityFlags, ...]
    passed: int
    failed: int

    def __post_init__(self) -> None:
        flags = self.flags
        if isinstance(flags, list):
            flags = tuple(flags)
            object.__setattr__(self, "flags", flags)
        if not isinstance(flags, tuple) or not all(
            isinstance(entry, QualityFlags) for entry in flags
        ):
            raise ValidationError("flags must be a tuple/list of QualityFlags")
        for name in ("passed", "failed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValidationError(f"{name} must be a non-negative int, got {value!r}")
        if self.passed + self.failed != len(flags):
            raise ValidationError("passed + failed must equal the flag count")

    def to_dict(self) -> dict[str, Any]:
        """JSON-native summary with a stable issue histogram."""
        histogram: dict[str, int] = {}
        for entry in self.flags:
            for issue in entry.issues:
                histogram[issue] = histogram.get(issue, 0) + 1
        return {
            "sample_count": len(self.flags),
            "passed": self.passed,
            "failed": self.failed,
            "issues": {key: histogram[key] for key in sorted(histogram)},
        }


def canonical_answer_prefix(language: str) -> str:
    """Literal prefix of the language's assistant template (derived, not copied)."""
    if language not in ASSISTANT_TEMPLATES:
        raise ValidationError(f"unknown language {language!r}")
    return ASSISTANT_TEMPLATES[language].text.split("{", 1)[0]


def check_sample(
    sample: InstructionSample,
    snr_db: Mapping[str, float] | None = None,
    thresholds: QualityThresholds | None = None,
) -> QualityFlags:
    """Flag one sample against ``thresholds`` (SNR sidecar optional)."""
    _require_sample(sample)
    cfg = thresholds if thresholds is not None else QualityThresholds()
    if not isinstance(cfg, QualityThresholds):
        raise ValidationError(f"thresholds must be a QualityThresholds, got {type(cfg).__name__}")
    if snr_db is not None and not isinstance(snr_db, Mapping):
        raise ValidationError(f"snr_db must be a mapping or None, got {type(snr_db).__name__}")
    issues: list[str] = []
    refs = [turn.audio for turn in sample.turns if turn.audio is not None]
    duration_ok = all(cfg.min_duration_ms <= ref.duration_ms <= cfg.max_duration_ms for ref in refs)
    if not duration_ok:
        issues.append(ISSUE_DURATION)
    snr_ok: bool | None = None
    if snr_db is not None and cfg.min_snr_db is not None:
        values = []
        for ref in refs:
            if ref.manifest_id in snr_db:
                field = f"snr for {ref.manifest_id!r}"
                values.append(_finite_number(field, snr_db[ref.manifest_id]))
        if values:
            snr_ok = all(value >= cfg.min_snr_db for value in values)
            if not snr_ok:
                issues.append(ISSUE_SNR)
    prefix = canonical_answer_prefix(sample.language)
    final = sample.turns[-1]
    answer_canonical = final.text is not None and final.text.startswith(prefix)
    if not answer_canonical:
        issues.append(ISSUE_ANSWER)
    return QualityFlags(
        sample_id=sample.sample_id,
        duration_ok=duration_ok,
        snr_ok=snr_ok,
        answer_canonical=answer_canonical,
        issues=tuple(issues),
    )


def check_samples(
    samples,
    snr_db: Mapping[str, float] | None = None,
    thresholds: QualityThresholds | None = None,
) -> QualityReport:
    """Flag a whole corpus and aggregate the verdicts."""
    items = _require_samples(samples)
    flags = tuple(check_sample(sample, snr_db, thresholds) for sample in items)
    passed = sum(1 for entry in flags if entry.ok)
    return QualityReport(flags=flags, passed=passed, failed=len(flags) - passed)


def _require_sample(sample) -> InstructionSample:
    if not isinstance(sample, InstructionSample):
        raise ValidationError(
            f"samples must contain InstructionSample, got {type(sample).__name__}"
        )
    return sample


def _require_samples(samples) -> list[InstructionSample]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, (list, tuple)):
        raise ValidationError(f"samples must be a list/tuple, got {type(samples).__name__}")
    items = [_require_sample(sample) for sample in samples]
    ids = [sample.sample_id for sample in items]
    if len(set(ids)) != len(ids):
        raise ValidationError("samples must have unique sample_ids")
    return items
