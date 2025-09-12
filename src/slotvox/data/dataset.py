"""Dataset assembly: split-aware generation with provenance hashes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from slotvox.config.generation import SPLITS, GenerationConfig
from slotvox.data.factory import GeneratedExample, render_example
from slotvox.data.patterns import templates_for
from slotvox.data.splits import check_policy, pattern_splits, speaker_splits
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


def generate_dataset(
    config: GenerationConfig | None = None,
    *,
    language: str = "zh",
    policy: str = "speaker",
    seed: int | None = None,
) -> GeneratedDataset:
    """Generate a labeled synthetic dataset with leakage-free splits.

    Deterministic: identical arguments produce identical labels, acoustics,
    and hashes. Speaker policy keeps voices disjoint across splits; pattern
    policy keeps phrasings disjoint (speakers may then repeat).
    """
    cfg = config if config is not None else GenerationConfig()
    if not isinstance(cfg, GenerationConfig):
        raise ValidationError(f"config must be a GenerationConfig, got {type(cfg).__name__}")
    domain = builtin_domain(cfg.domain)
    check_policy(policy)
    if language not in LANGUAGES:
        raise ValidationError(f"language must be one of {LANGUAGES}, got {language!r}")
    base_seed = cfg.seed if seed is None else seed
    make_rng(base_seed)
    templates = templates_for(cfg.domain, language)
    non_empty = [split for split in SPLITS if cfg.counts.get(split, 0) > 0]

    split_speakers: dict[str, tuple[int, ...]] = {}
    split_templates: dict[str, tuple[Any, ...]] = {}
    if policy == "speaker":
        buckets = speaker_splits(cfg.n_speakers, cfg.counts, base_seed)
        for split in non_empty:
            split_speakers[split] = tuple(
                sorted(speaker for speaker, target in buckets.items() if target == split)
            )
            split_templates[split] = templates
    else:
        buckets_p = pattern_splits(
            [template.pattern_id for template in templates], cfg.counts, base_seed
        )
        for split in non_empty:
            wanted = {pid for pid, target in buckets_p.items() if target == split}
            split_templates[split] = tuple(t for t in templates if t.pattern_id in wanted)
            split_speakers[split] = tuple(range(cfg.n_speakers))

    examples: list[GeneratedExample] = []
    for split in SPLITS:
        count = cfg.counts.get(split, 0)
        if count == 0:
            continue
        speakers = split_speakers[split]
        split_tpls = split_templates[split]
        for index in range(count):
            template = split_tpls[index % len(split_tpls)]
            speaker_id = speakers[index % len(speakers)]
            snr = float(cfg.noise_snr_db[index % len(cfg.noise_snr_db)])
            examples.append(
                render_example(
                    domain,
                    template,
                    language,
                    utterance_id=f"{split}-{index:05d}",
                    split=split,
                    speaker_id=speaker_id,
                    snr_db=snr,
                    seed=_example_seed(base_seed, split, index),
                )
            )
    return GeneratedDataset(
        domain_id=cfg.domain,
        language=language,
        policy=policy,
        seed=base_seed,
        config=cfg,
        examples=tuple(examples),
    )
