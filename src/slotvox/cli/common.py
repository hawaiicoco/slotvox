"""Shared CLI plumbing: exit codes, output, and error mapping.

Exit codes are a contract:

- ``0`` success
- ``2`` usage or validation error (bad flags, configs, ids, counts)
- ``3`` data or schema error (corrupt artifacts, unknown schemas, audio)
- ``4`` runtime protocol error (adapter or streaming failures)
- ``1`` unexpected crash (a bug; the traceback goes to stderr)

``SchemaError``/``AudioError``/``ConfigError`` subclass
``ValidationError``, so the mapping checks them first.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any

from slotvox.errors import (
    AdapterError,
    AudioError,
    ConfigError,
    SchemaError,
    StreamingError,
    ValidationError,
)

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_USAGE = 2
EXIT_DATA = 3
EXIT_RUNTIME = 4


def emit(args: Any, payload: dict[str, Any], lines: list[str]) -> None:
    """Print canonical JSON (with ``--json``) or the human-readable lines."""
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        for line in lines:
            print(line)


def run_command(handler: Callable[[Any], int], args: Any) -> int:
    """Run one command handler, mapping slotvox errors onto exit codes."""
    try:
        return handler(args)
    except (SchemaError, AudioError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DATA
    except (ConfigError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except (AdapterError, StreamingError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME
