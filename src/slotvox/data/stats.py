"""Dataset statistics summaries with provenance.

``summarize`` produces a canonical-JSON-ready dict describing a generated
dataset: per-split counts, intent/label distributions, speaker and pattern
coverage, noise levels, and durations. Every number is computed from the
examples actually present — summaries never extrapolate.
"""

from __future__ import annotations

from typing import Any

from slotvox.config.generation import SPLITS
from slotvox.data.dataset import GeneratedDataset
from slotvox.errors import ValidationError
from slotvox.tagging.bio import OUTSIDE

STATS_SCHEMA_ID = "slotvox.dataset-stats"
STATS_SCHEMA_VERSION = 1


def _round(value: float, digits: int) -> float:
    return round(float(value), digits)


def summarize(dataset: GeneratedDataset) -> dict[str, Any]:
    """Compute the full statistics envelope for ``dataset``."""
    if not isinstance(dataset, GeneratedDataset):
        raise ValidationError(f"summarize expects a GeneratedDataset, got {type(dataset).__name__}")
    counts: dict[str, int] = {}
    speakers: dict[str, list[int]] = {}
    patterns: dict[str, list[str]] = {}
    snr_levels: dict[str, list[float]] = {}
    by_intent: dict[str, dict[str, int]] = {}
    slot_mentions: dict[str, int] = {}
    token_lengths: list[int] = []
    durations: list[float] = []
    for split in SPLITS:
        examples = dataset.split_examples(split)
        counts[split] = len(examples)
        speakers[split] = sorted({example.speaker_id for example in examples})
        patterns[split] = sorted({example.pattern_id for example in examples})
        snr_levels[split] = sorted(
            {example.snr_db for example in examples if example.snr_db is not None}
        )
        intents: dict[str, int] = {}
        for example in examples:
            intent = example.annotation.intent
            intents[intent] = intents.get(intent, 0) + 1
            token_lengths.append(len(example.annotation.tokens))
            durations.append(example.utterance.duration_s)
            for tag in example.annotation.tags:
                if tag != OUTSIDE and tag.startswith("B-"):
                    slot = tag[2:]
                    slot_mentions[slot] = slot_mentions.get(slot, 0) + 1
        by_intent[split] = {intent: intents[intent] for intent in sorted(intents)}
    total = len(dataset.examples)
    return {
        "schema": STATS_SCHEMA_ID,
        "schema_version": STATS_SCHEMA_VERSION,
        "domain_id": dataset.domain_id,
        "language": dataset.language,
        "policy": dataset.policy,
        "seed": dataset.seed,
        "counts": counts,
        "speakers": speakers,
        "patterns": patterns,
        "snr_levels": snr_levels,
        "by_intent": by_intent,
        "slot_mentions": {slot: slot_mentions[slot] for slot in sorted(slot_mentions)},
        "token_length": {
            "min": min(token_lengths) if token_lengths else 0,
            "max": max(token_lengths) if token_lengths else 0,
            "mean": _round(sum(token_lengths) / total, 4) if total else 0.0,
        },
        "duration_s": {
            "total": _round(sum(durations), 6) if durations else 0.0,
            "mean": _round(sum(durations) / total, 6) if total else 0.0,
        },
        "dataset_hash": dataset.dataset_hash,
    }
