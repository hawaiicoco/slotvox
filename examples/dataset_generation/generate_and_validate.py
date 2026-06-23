#!/usr/bin/env python3
"""Generate a synthetic SLU dataset, validate it, and summarize it.

Runs fully offline on synthetic signals — no real speech, no downloads.
The segments of every utterance map to intents and slots BY
CONSTRUCTION; this example shows the schema, the leak-free splits, and
the persistence round-trip.

Usage:
    python examples/dataset_generation/generate_and_validate.py --out DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

from slotvox.config import GenerationConfig
from slotvox.data.dataset import generate_dataset
from slotvox.data.persist import load_dataset, save_dataset
from slotvox.data.stats import summarize
from slotvox.schema.builtins import builtin_domain


def main() -> int:
    """Generate, validate, save, reload, and summarize one dataset."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--domain", default="weather")
    parser.add_argument("--language", default="zh", choices=("zh", "en"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    domain = builtin_domain(args.domain)
    print(f"domain:  {domain.domain} — {domain.description}")
    print(f"slots:   {', '.join(domain.slot_names)}")
    print(f"intents: {', '.join(domain.intent_names)}")

    config = GenerationConfig(
        domain=args.domain,
        counts={"train": 12, "dev": 4, "test": 4},
        n_speakers=4,
        seed=args.seed,
    )
    dataset = generate_dataset(config, language=args.language)
    print(f"generated {len(dataset.examples)} utterances {dataset.split_counts}")

    speakers = dataset.speakers_per_split()
    patterns = dataset.patterns_per_split()
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if dataset.policy == "speaker":
            assert not set(speakers[left]) & set(speakers[right]), "speaker leak"
        else:
            assert not set(patterns[left]) & set(patterns[right]), "pattern leak"
    shared_patterns = len(set(patterns["train"]) & set(patterns["dev"]))
    print(f"split check ({dataset.policy} policy): no leaks across train/dev/test")
    print(f"  (patterns shared across splits are expected under 'speaker': {shared_patterns} here)")

    root = save_dataset(dataset, args.out / "dataset", overwrite=True)
    reloaded = load_dataset(root)
    assert reloaded.dataset_hash == dataset.dataset_hash, "persistence changed the dataset"
    print(f"saved + reloaded; dataset hash {dataset.dataset_hash}")

    stats = summarize(reloaded)
    print(f"slot mentions: {stats['slot_mentions']}")
    print(f"mean duration: {stats['duration_s']['mean']:.3f} s")

    example = dataset.examples[0]
    print("first example:")
    print(f"  text:   {example.annotation.text}")
    print(f"  intent: {example.annotation.intent}")
    print(f"  tags:   {' '.join(example.annotation.tags)}")
    print("note: synthetic signals only — segments map to slots by construction,")
    print("      this is NOT real speech and no quality claim is implied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
