"""``slotvox export-instructions`` — dataset to instruction corpus.

Torch-free: renders every saved example into an interleaved speech-text
instruction sample (audio referenced by manifest id), runs the quality
flags with the SNR sidecar from the dataset itself, and writes the
versioned corpus export.
"""

from __future__ import annotations

from typing import Any

from slotvox.cli.common import EXIT_OK, emit
from slotvox.data.persist import load_dataset
from slotvox.instructions.export import export_instructions, read_export_manifest
from slotvox.instructions.quality import QualityThresholds
from slotvox.instructions.templates import render_sample


def register_export_instructions_command(subparsers: Any) -> None:
    """Attach the ``export-instructions`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser(
        "export-instructions", help="export a saved dataset as an instruction corpus"
    )
    parser.add_argument("--dataset", required=True, help="dataset directory")
    parser.add_argument("--out", required=True, help="corpus output directory")
    parser.add_argument(
        "--min-snr-db",
        type=float,
        default=None,
        help="flag samples whose SNR sidecar value falls below this",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(handler=cmd_export_instructions)


def cmd_export_instructions(args: Any) -> int:
    """Render, quality-flag, and export one dataset as instructions."""
    dataset = load_dataset(args.dataset)
    samples = [
        render_sample(example, domain=dataset.domain_id, dataset_hash=dataset.dataset_hash)
        for example in dataset.examples
    ]
    snr_db = {
        f"{dataset.domain_id}-{example.utterance_id}": example.snr_db
        for example in dataset.examples
        if example.snr_db is not None
    }
    thresholds = (
        QualityThresholds(min_snr_db=args.min_snr_db) if args.min_snr_db is not None else None
    )
    root = export_instructions(
        samples, args.out, snr_db=snr_db, thresholds=thresholds, overwrite=args.overwrite
    )
    manifest = read_export_manifest(root)
    quality = manifest["quality"]
    payload = {
        "path": str(root),
        "sample_count": manifest["sample_count"],
        "quality": quality,
        "content_hash": manifest["content_hash"],
    }
    lines = [
        f"exported {manifest['sample_count']} instruction samples to {root}",
        f"quality: {quality['passed']} passed, {quality['failed']} failed",
        f"corpus hash: {manifest['content_hash']}",
    ]
    emit(args, payload, lines)
    return EXIT_OK
