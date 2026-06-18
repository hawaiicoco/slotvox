"""``slotvox report`` — render Markdown/HTML reports from a run envelope."""

from __future__ import annotations

from typing import Any

from slotvox.cli.common import EXIT_OK, emit
from slotvox.eval.report import (
    html_from_envelope,
    markdown_from_envelope,
    write_report_files,
)
from slotvox.eval.runs import validate_run_envelope
from slotvox.schema.serialize import read_json


def register_report_command(subparsers: Any) -> None:
    """Attach the ``report`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser("report", help="render reports from an eval run envelope")
    parser.add_argument("--run", required=True, help="run envelope JSON file")
    parser.add_argument("--out", required=True, help="report output directory")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(handler=cmd_report)


def cmd_report(args: Any) -> int:
    """Validate one envelope and write both report formats."""
    validated = validate_run_envelope(read_json(args.run))
    markdown = markdown_from_envelope(validated)
    html_text = html_from_envelope(validated)
    root = write_report_files(markdown, html_text, args.out, overwrite=args.overwrite)
    payload = {
        "path": str(root),
        "run_id": validated["run_id"],
        "files": ["report.html", "report.md"],
    }
    lines = [
        f"reports for run {validated['run_id']} written to {root}",
        f"  {root / 'report.md'}",
        f"  {root / 'report.html'}",
    ]
    emit(args, payload, lines)
    return EXIT_OK
