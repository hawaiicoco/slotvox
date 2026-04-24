"""Argument parsing and dispatch for the slotvox CLI.

Subcommands register themselves on the shared parser; every command runs
fully offline and returns an exit code under the documented contract in
:mod:`slotvox.cli.common`.
"""

from __future__ import annotations

import argparse
import sys

from slotvox._version import __version__
from slotvox.cli.common import EXIT_USAGE, run_command
from slotvox.cli.schema import register_schema_command


def build_parser() -> argparse.ArgumentParser:
    """The full command-line parser."""
    parser = argparse.ArgumentParser(
        prog="slotvox",
        description=(
            "End-to-end spoken language understanding on synthetic speech. "
            "Every subcommand runs fully offline."
        ),
    )
    parser.add_argument("--version", action="version", version=f"slotvox {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    register_schema_command(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_usage(sys.stderr)
        print("error: a COMMAND is required", file=sys.stderr)
        return EXIT_USAGE
    return run_command(handler, args)
