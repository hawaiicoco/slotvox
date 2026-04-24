"""``slotvox schema`` — inspect the built-in task domains."""

from __future__ import annotations

from typing import Any

from slotvox.cli.common import EXIT_OK, emit
from slotvox.schema.builtins import BUILTIN_DOMAIN_IDS, builtin_domain


def register_schema_command(subparsers: Any) -> None:
    """Attach the ``schema`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser("schema", help="print built-in domain schemas")
    parser.add_argument("domains", nargs="*", help="domain ids (default: all built-ins)")
    parser.add_argument("--list", action="store_true", help="list domain ids and exit")
    parser.add_argument("--json", action="store_true", help="canonical JSON output")
    parser.set_defaults(handler=cmd_schema)


def cmd_schema(args: Any) -> int:
    """Print domain ids (``--list``) or full domain schemas."""
    if args.list:
        for domain_id in BUILTIN_DOMAIN_IDS:
            print(domain_id)
        return EXIT_OK
    domain_ids = tuple(args.domains) or BUILTIN_DOMAIN_IDS
    specs = [builtin_domain(domain_id) for domain_id in domain_ids]
    payload = {"domains": [spec.to_dict() for spec in specs]}
    lines = []
    for spec in specs:
        lines.append(f"{spec.domain}: {spec.description}")
        lines.append(f"  slots:   {', '.join(spec.slot_names)}")
        lines.append(f"  intents: {', '.join(spec.intent_names)}")
    emit(args, payload, lines)
    return EXIT_OK
