"""``slotvox serve-mock`` — foreground loopback mock of the protocol.

Binds ``127.0.0.1`` only and serves saved fixtures through the same
:class:`ReplayAdapter` the tests use. ``--requests N`` exits after N
requests (demos and CI stay bounded); ``--requests 0`` serves until
interrupted. Fully offline; validates protocol behavior, not model
quality.
"""

from __future__ import annotations

import time
from typing import Any

from slotvox.adapters.mock_server import MockInferServer
from slotvox.adapters.replay import ReplayAdapter, load_fixtures
from slotvox.cli.common import EXIT_OK, emit
from slotvox.errors import StreamingError, ValidationError


def register_serve_mock_command(subparsers: Any) -> None:
    """Attach the ``serve-mock`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser("serve-mock", help="serve fixtures over loopback HTTP")
    parser.add_argument("--fixtures", required=True, help="fixture store JSON file")
    parser.add_argument(
        "--requests", type=int, default=1, help="exit after N requests (0 = until interrupted)"
    )
    parser.add_argument(
        "--timeout-s", type=float, default=60.0, help="max seconds to wait for the budget"
    )
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(handler=cmd_serve_mock)


def cmd_serve_mock(args: Any) -> int:
    """Serve fixtures until the request budget or timeout is reached."""
    if args.requests < 0:
        raise ValidationError(f"--requests must be >= 0, got {args.requests!r}")
    if args.timeout_s <= 0:
        raise ValidationError(f"--timeout-s must be positive, got {args.timeout_s!r}")
    fixtures = load_fixtures(args.fixtures)
    with MockInferServer(ReplayAdapter(fixtures)) as server:
        print(f"serving at {server.base_url}", flush=True)
        if args.requests == 0:
            try:
                while True:
                    time.sleep(0.2)
            except KeyboardInterrupt:  # pragma: no cover - interactive path
                pass
        else:
            deadline = time.monotonic() + args.timeout_s
            while server.requests_served < args.requests:
                if time.monotonic() > deadline:
                    raise StreamingError(
                        f"timed out waiting for {args.requests} requests "
                        f"(served {server.requests_served})"
                    )
                time.sleep(0.05)
        served = server.requests_served
        base_url = server.base_url
    payload = {"base_url": base_url, "requests_served": served, "fixtures": len(fixtures)}
    lines = [f"served {served} request(s) from {len(fixtures)} fixture(s); server stopped"]
    emit(args, payload, lines)
    return EXIT_OK
