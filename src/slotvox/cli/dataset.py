"""``slotvox generate`` / ``slotvox featurize`` — synthetic data pipelines."""

from __future__ import annotations

from typing import Any

from slotvox.cli.common import EXIT_OK, emit
from slotvox.config import GenerationConfig
from slotvox.config.generation import SPLITS
from slotvox.data.dataset import generate_dataset
from slotvox.data.persist import save_dataset
from slotvox.data.splits import SPLIT_POLICIES
from slotvox.errors import ValidationError


def parse_counts(text: str) -> dict[str, int]:
    """Parse ``"train=8,dev=2"`` strictly into a counts dict."""
    if not isinstance(text, str) or not text:
        raise ValidationError("counts must be a non-empty string like 'train=8,dev=2'")
    counts: dict[str, int] = {}
    for part in text.split(","):
        if "=" not in part:
            raise ValidationError(f"count entry {part!r} must look like 'split=N'")
        key, _, raw = part.partition("=")
        if key not in SPLITS:
            raise ValidationError(f"unknown split {key!r}; expected one of {SPLITS}")
        if key in counts:
            raise ValidationError(f"duplicate split {key!r} in counts")
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValidationError(f"count for {key!r} must be an int, got {raw!r}") from exc
        if value < 0:
            raise ValidationError(f"count for {key!r} must be >= 0, got {value}")
        counts[key] = value
    return counts


def register_generate_command(subparsers: Any) -> None:
    """Attach the ``generate`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser("generate", help="generate a synthetic dataset")
    parser.add_argument("--domain", default="weather", help="built-in domain id")
    parser.add_argument("--language", default="zh", choices=("zh", "en"))
    parser.add_argument("--policy", default="speaker", choices=SPLIT_POLICIES)
    parser.add_argument(
        "--seed", type=int, default=None, help="dataset seed (config default when omitted)"
    )
    parser.add_argument("--counts", default="train=64,dev=16,test=16")
    parser.add_argument("--speakers", type=int, default=4)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(handler=cmd_generate)


def register_dataset_commands(subparsers: Any) -> None:
    """Attach the dataset subcommands to ``subparsers``."""
    register_generate_command(subparsers)


def cmd_generate(args: Any) -> int:
    """Generate a dataset and save it in the slotvox.dataset format."""
    counts = parse_counts(args.counts)
    kwargs: dict[str, Any] = {}
    if args.seed is not None:
        kwargs["seed"] = args.seed
    config = GenerationConfig(domain=args.domain, counts=counts, n_speakers=args.speakers, **kwargs)
    dataset = generate_dataset(config, language=args.language, policy=args.policy, seed=args.seed)
    root = save_dataset(dataset, args.out, overwrite=args.overwrite)
    payload = {
        "domain": dataset.domain_id,
        "language": dataset.language,
        "policy": dataset.policy,
        "counts": dataset.split_counts,
        "dataset_hash": dataset.dataset_hash,
        "path": str(root),
    }
    nonzero = ", ".join(
        f"{split}={count}" for split, count in sorted(dataset.split_counts.items()) if count
    )
    lines = [
        f"generated {sum(dataset.split_counts.values())} examples ({nonzero})",
        f"dataset hash: {dataset.dataset_hash}",
        f"written to:   {root}",
    ]
    emit(args, payload, lines)
    return EXIT_OK
