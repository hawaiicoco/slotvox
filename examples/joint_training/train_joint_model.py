#!/usr/bin/env python3
"""Train the joint intent-slot model on synthetic data (torch extra, CPU).

Deterministic given --seed. Honest scope: the metric movement below
happens on SYNTHETIC factory signals with a small from-scratch model;
it is not a benchmark and says nothing about real speech.

Usage:
    python examples/joint_training/train_joint_model.py --out DIR
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


def main() -> int:
    """Train from scratch on synthetic data; print before/after metrics."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--domain", default="music-control")
    parser.add_argument("--language", default="zh", choices=("zh", "en"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "this example requires the torch extra: pip install 'slotvox[torch]'",
            file=sys.stderr,
        )
        return 2

    from slotvox.config import FeatureConfig, GenerationConfig, TrainConfig
    from slotvox.data.dataset import generate_dataset
    from slotvox.data.features_store import featurize_dataset
    from slotvox.data.persist import save_dataset
    from slotvox.models.checkpoint import save_checkpoint
    from slotvox.models.dataset import encode_split
    from slotvox.models.joint import JointIntentSlotModel, ModelSpec
    from slotvox.models.train import evaluate, train_model
    from slotvox.models.vocab import Vocab, build_intent_vocab, build_tag_vocab
    from slotvox.schema.builtins import builtin_domain

    domain = builtin_domain(args.domain)
    feature_config = FeatureConfig(n_mels=24)
    train_config = TrainConfig(
        encoder="tcn",
        hidden_size=32,
        layers=2,
        epochs=args.epochs,
        batch_size=8,
        learning_rate=3e-3,
        seed=args.seed,
    )
    spec = ModelSpec.from_domain(domain, feature_config, train_config)
    config = GenerationConfig(
        domain=args.domain, counts={"train": 48, "dev": 12}, n_speakers=6, seed=args.seed
    )
    dataset = generate_dataset(config, language=args.language)
    print(f"dataset: {len(dataset.examples)} synthetic utterances {dataset.split_counts}")

    with tempfile.TemporaryDirectory(prefix="slotvox-example-") as tmp:
        data_dir = Path(tmp) / "dataset"
        save_dataset(dataset, data_dir)
        feat_dir = featurize_dataset(dataset, feature_config, Path(tmp) / "features")
        intents = Vocab(build_intent_vocab(domain))
        tags = Vocab(build_tag_vocab(domain))
        train_rows = encode_split(data_dir, feat_dir, "train", intents=intents, tags=tags)
        dev_rows = encode_split(data_dir, feat_dir, "dev", intents=intents, tags=tags)

        model = JointIntentSlotModel(spec)
        before = evaluate(model, dev_rows)
        print(
            f"RESULT before loss={before[0]:.4f} "
            f"intent_acc={before[1]:.4f} frame_acc={before[2]:.4f}"
        )
        history = train_model(model, train_rows, dev_rows)
        after = evaluate(model, dev_rows)
        print(
            f"RESULT after loss={after[0]:.4f} intent_acc={after[1]:.4f} frame_acc={after[2]:.4f}"
        )

        print("epoch | train loss | dev loss | dev intent acc | dev frame acc")
        for record in history:
            print(
                f"{record.epoch:5d} | {record.train_loss:10.4f} | {record.dev_loss:8.4f} "
                f"| {record.dev_intent_accuracy:14.4f} | {record.dev_slot_frame_accuracy:13.4f}"
            )

        args.out.mkdir(parents=True, exist_ok=True)
        checkpoint = save_checkpoint(
            args.out / "model.ckpt", model, epoch=len(history), history=history
        )

    print(f"checkpoint: {checkpoint}")
    print(
        "note: synthetic data only — the dev movement above is real for THIS "
        "seed and dataset; it is not a benchmark and does not transfer to real speech."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
