"""``slotvox eval`` — score a checkpoint over one split of a saved dataset.

Gold frame tags come from the by-construction alignment
(:func:`slotvox.data.alignment.frame_span_tags`); predictions are the
model's greedy decode with ``promote`` repair, so both sides are
structurally valid BIO. The output is a versioned run envelope. Every
number describes THIS synthetic run — no real-world quality claim.
Requires the torch extra.
"""

from __future__ import annotations

from typing import Any

from slotvox.cli.common import EXIT_OK, emit
from slotvox.errors import ValidationError


def register_eval_command(subparsers: Any) -> None:
    """Attach the ``eval`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser(
        "eval", help="score a checkpoint over one dataset split (torch extra)"
    )
    parser.add_argument("--dataset", required=True, help="dataset directory")
    parser.add_argument("--model", required=True, help="checkpoint file")
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--run-id", default="cli-eval")
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--out", default=None, help="optional run-envelope JSON file")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(handler=cmd_eval)


def cmd_eval(args: Any) -> int:
    """Decode one split, score every metric, and emit the run envelope."""
    try:
        import torch
    except ImportError as exc:
        raise ValidationError(
            "eval requires the torch extra: pip install 'slotvox[torch]'"
        ) from exc

    from slotvox.data.alignment import frame_span_tags, frame_token_map
    from slotvox.data.persist import load_dataset
    from slotvox.eval.metrics import TurnRecord, slice_pairs
    from slotvox.eval.runs import ScoredRun
    from slotvox.features.frontend import log_mel
    from slotvox.models.checkpoint import load_checkpoint
    from slotvox.schema.serialize import write_json_atomic
    from slotvox.tagging.bio import repair_sequence

    loaded = load_checkpoint(args.model)
    dataset = load_dataset(args.dataset)
    if dataset.domain_id != loaded.spec.domain_id:
        raise ValidationError(
            f"dataset domain {dataset.domain_id!r} does not match model domain "
            f"{loaded.spec.domain_id!r}"
        )
    examples = dataset.split_examples(args.split)
    if not examples:
        raise ValidationError(f"split {args.split!r} of the dataset is empty")
    feature_config = loaded.spec.feature_config
    model = loaded.model
    model.eval()
    records = []
    for example in examples:
        mel = log_mel(example.utterance.samples, feature_config)
        token_map = frame_token_map(example.utterance, feature_config)
        gold = frame_span_tags(example.annotation.tags, token_map)
        intent, pred = model.predict_greedy(torch.from_numpy(mel))
        noise = "clean" if example.snr_db is None else f"snr-{int(example.snr_db)}"
        records.append(
            TurnRecord(
                example.annotation.intent,
                intent,
                gold,
                repair_sequence(pred, "promote"),
                slices=slice_pairs({"noise": noise, "speaker": f"spk-{example.speaker_id}"}),
            )
        )
    run = ScoredRun.evaluate(
        args.run_id,
        records,
        metadata={
            "model_hash": loaded.spec.model_hash,
            "dataset_hash": dataset.dataset_hash,
            "split": args.split,
        },
        bootstrap_seed=args.bootstrap_seed,
        replicates=args.replicates,
    )
    if args.out:
        write_json_atomic(args.out, run.to_dict())
    payload = {
        "run_id": run.run_id,
        "records": run.record_count,
        "intent_accuracy": run.intent.accuracy,
        "slot_f1_strict": run.slot_strict.f1,
        "slot_f1_partial": run.slot_partial.f1,
        "turn_accuracy": run.turn.accuracy,
        "envelope": str(args.out) if args.out else None,
    }
    lines = [
        f"evaluated {run.record_count} turns from split {args.split!r} "
        "(synthetic data, greedy decoding)",
        f"intent accuracy:   {payload['intent_accuracy']:.4f}",
        f"slot f1 (strict):  {payload['slot_f1_strict']:.4f}",
        f"slot f1 (partial): {payload['slot_f1_partial']:.4f}",
        f"turn accuracy:     {payload['turn_accuracy']:.4f}",
    ]
    if args.out:
        lines.append(f"envelope: {args.out}")
    emit(args, payload, lines)
    return EXIT_OK
