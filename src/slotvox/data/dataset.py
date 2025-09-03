"""Dataset assembly: split-aware generation with provenance hashes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from slotvox.config.generation import SPLITS, GenerationConfig
from slotvox.data.factory import GeneratedExample
from slotvox.data.splits import check_policy
from slotvox.errors import ValidationError
from slotvox.schema.annotations import LANGUAGES
from slotvox.schema.builtins import builtin_domain
from slotvox.util.jsoncanon import stable_hash
from slotvox.util.seed import derive_seed, make_rng

DATASET_SCHEMA_ID = "slotvox.dataset-provenance"
DATASET_SCHEMA_VERSION = 1


def _example_seed(base_seed: int, split: str, index: int) -> int:
    """Per-example seed derived deterministically from the dataset seed."""
    return derive_seed("example", base_seed, split, index)


@dataclass(frozen=True)
class GeneratedDataset:
    """A fully generated dataset with provenance."""

    domain_id: str
    language: str
    policy: str
    seed: int
    config: GenerationConfig
    examples: tuple[GeneratedExample, ...]

    def __post_init__(self) -> None:
        builtin_domain(self.domain_id)  # strict: unknown domains rejected
        if self.language not in LANGUAGES:
            raise ValidationError(f"language must be one of {LANGUAGES}, got {self.language!r}")
        check_policy(self.policy)
        make_rng(self.seed)
        if not isinstance(self.config, GenerationConfig):
            raise ValidationError(
                f"config must be a GenerationConfig, got {type(self.config).__name__}"
            )
        if self.config.domain != self.domain_id:
            raise ValidationError(
                f"config domain {self.config.domain!r} != dataset domain {self.domain_id!r}"
            )
        examples = self.examples
        if isinstance(examples, list):
            examples = tuple(examples)
            object.__setattr__(self, "examples", examples)
        if not isinstance(examples, tuple):
            raise ValidationError("examples must be a tuple/list of GeneratedExample")
        seen_ids: set[str] = set()
        per_split: dict[str, int] = {split: 0 for split in SPLITS}
        for example in examples:
            if not isinstance(example, GeneratedExample):
                raise ValidationError("examples must contain GeneratedExample instances")
            if example.utterance_id in seen_ids:
                raise ValidationError(f"duplicate utterance_id {example.utterance_id!r}")
            seen_ids.add(example.utterance_id)
            per_split[example.split] += 1
        expected = {split: self.config.counts.get(split, 0) for split in SPLITS}
        if per_split != expected:
            raise ValidationError(
                f"examples per split {per_split} do not match config counts {expected}"
            )

    def split_examples(self, split: str) -> tuple[GeneratedExample, ...]:
        """All examples of one split (strict)."""
        if split not in SPLITS:
            raise ValidationError(f"split must be one of {SPLITS}, got {split!r}")
        return tuple(example for example in self.examples if example.split == split)

    @property
    def split_counts(self) -> dict[str, int]:
        """Example count per split."""
        return {split: len(self.split_examples(split)) for split in SPLITS}

    def speakers_per_split(self) -> dict[str, tuple[int, ...]]:
        """Sorted unique speaker ids per split (for leakage assertions)."""
        return {
            split: tuple(sorted({example.speaker_id for example in self.split_examples(split)}))
            for split in SPLITS
        }

    def patterns_per_split(self) -> dict[str, tuple[str, ...]]:
        """Sorted unique pattern ids per split (for leakage assertions)."""
        return {
            split: tuple(sorted({example.pattern_id for example in self.split_examples(split)}))
            for split in SPLITS
        }

    def provenance(self) -> dict[str, Any]:
        """Provenance envelope: everything needed to regenerate the dataset."""
        return {
            "schema": DATASET_SCHEMA_ID,
            "schema_version": DATASET_SCHEMA_VERSION,
            "domain_id": self.domain_id,
            "language": self.language,
            "policy": self.policy,
            "seed": self.seed,
            "config": self.config.to_dict(),
        }

    @property
    def dataset_hash(self) -> str:
        """Stable hash over provenance and every example's labels."""
        fingerprints = [
            {
                "utterance_id": example.utterance_id,
                "pattern_id": example.pattern_id,
                "speaker_id": example.speaker_id,
                "snr_db": example.snr_db,
                "annotation": example.annotation.to_dict(),
                "slot_values": {slot: list(v) for slot, v in example.slot_values.items()},
            }
            for example in self.examples
        ]
        return stable_hash({"provenance": self.provenance(), "examples": fingerprints})
