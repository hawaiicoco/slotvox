"""``slotvox train`` — train the joint model on a freshly generated dataset.

The dataset and its features are derived data: they are generated into a
temporary directory, encoded, trained on, and discarded — only the
checkpoint, spec, and history land in ``--out``. Training is fully
deterministic given ``--seed`` (CPU, single-threaded, seeded shuffles).
Requires the torch extra.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from slotvox.cli.common import EXIT_OK, emit
from slotvox.errors import ValidationError


def register_train_command(subparsers: Any) -> None:
    """Attach the ``train`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser("train", help="train the joint intent-slot model (torch extra)")
    parser.add_argument("--domain", default="weather", help="built-in domain id")
    parser.add_argument("--language", default="zh", choices=("zh", "en"))
    parser.add_argument("--counts", default="train=32,dev=8")
    parser.add_argument("--speakers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--encoder", default="tcn", choices=("tcn", "gru"))
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--n-mels", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--out", required=True, help="checkpoint output directory")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(handler=cmd_train)


def cmd_train(args: Any) -> int:
    """Generate, featurize, train, and checkpoint — deterministic per seed."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ValidationError(
            "train requires the torch extra: pip install 'slotvox[torch]'"
        ) from exc

    from slotvox.cli.dataset import parse_counts
    from slotvox.config import FeatureConfig, GenerationConfig, TrainConfig
    from slotvox.data.dataset import generate_dataset
    from slotvox.data.features_store import featurize_dataset
    from slotvox.data.persist import save_dataset
    from slotvox.models.checkpoint import save_checkpoint
    from slotvox.models.dataset import encode_split
    from slotvox.models.joint import JointIntentSlotModel, ModelSpec
    from slotvox.models.train import train_model
    from slotvox.models.vocab import Vocab, build_intent_vocab, build_tag_vocab
    from slotvox.schema.builtins import builtin_domain
    from slotvox.schema.serialize import write_json_atomic

    out = Path(args.out)
    if (out / "model.ckpt").exists() and not args.overwrite:
        raise ValidationError(f"{out} already holds a trained model; pass --overwrite to replace")
    counts = parse_counts(args.counts)
    config = GenerationConfig(
        domain=args.domain, counts=counts, n_speakers=args.speakers, seed=args.seed
    )
    feature_config = FeatureConfig(n_mels=args.n_mels)
    domain = builtin_domain(args.domain)
    spec = ModelSpec.from_domain(
        domain,
        feature_config,
        TrainConfig(
            encoder=args.encoder,
            hidden_size=args.hidden,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        ),
    )
    dataset = generate_dataset(config, language=args.language)
    with tempfile.TemporaryDirectory(prefix="slotvox-train-") as tmp:
        data_dir = Path(tmp) / "dataset"
        save_dataset(dataset, data_dir)
        feat_dir = featurize_dataset(dataset, feature_config, Path(tmp) / "features")
        intents = Vocab(build_intent_vocab(domain))
        tags = Vocab(build_tag_vocab(domain))
        train_rows = encode_split(data_dir, feat_dir, "train", intents=intents, tags=tags)
        dev_rows = encode_split(data_dir, feat_dir, "dev", intents=intents, tags=tags)
        model = JointIntentSlotModel(spec)
        history = train_model(model, train_rows, dev_rows)
    out.mkdir(parents=True, exist_ok=True)
    final = history[-1] if history else None
    metrics = (
        None
        if final is None
        else {
            "final_train_loss": final.train_loss,
            "final_dev_loss": final.dev_loss,
            "final_dev_intent_accuracy": final.dev_intent_accuracy,
            "final_dev_slot_frame_accuracy": final.dev_slot_frame_accuracy,
        }
    )
    save_checkpoint(out / "model.ckpt", model, epoch=len(history), history=history, metrics=metrics)
    write_json_atomic(out / "history.json", [record.to_dict() for record in history])
    write_json_atomic(out / "spec.json", spec.to_dict())
    payload = {
        "checkpoint": str(out / "model.ckpt"),
        "epochs": len(history),
        "model_hash": spec.model_hash,
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "final_train_loss": None if final is None else final.train_loss,
        "final_dev_loss": None if final is None else final.dev_loss,
        "final_dev_intent_accuracy": None if final is None else final.dev_intent_accuracy,
    }
    lines = [
        f"trained {len(history)} epoch(s) on {len(train_rows)} examples "
        f"({args.domain}, {args.language}, encoder {args.encoder})",
        f"final train loss: {payload['final_train_loss']}",
        f"final dev loss:   {payload['final_dev_loss']}",
        f"dev intent acc:   {payload['final_dev_intent_accuracy']}",
        f"checkpoint:       {out / 'model.ckpt'}",
        "note: synthetic data only; these numbers describe this run, not real speech.",
    ]
    emit(args, payload, lines)
    return EXIT_OK
