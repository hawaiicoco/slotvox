#!/usr/bin/env python3
"""Smoke-test a built slotvox wheel by installing it into a throwaway venv.

Catches packaging mistakes that in-tree tests cannot see: missing modules,
broken console-script entry points, or undeclared runtime dependencies.

Usage:
    make release                       # build sdist + wheel, then this check
    python scripts/check_package.py    # check the newest wheel in dist/
    python scripts/check_package.py --wheel dist/slotvox-0.1.0-py3-none-any.whl
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

IMPORT_CHECK = (
    "import re, slotvox;"
    "assert re.fullmatch(r'\\d+\\.\\d+\\.\\d+', slotvox.__version__);"
    "print(slotvox.__version__)"
)


def _version_key(wheel: Path) -> tuple[int, ...]:
    match = re.match(r"slotvox-(\d+)\.(\d+)\.(\d+)", wheel.name)
    if match is None:
        return (-1, -1, -1)
    return tuple(int(group) for group in match.groups())


def find_wheel(explicit: str | None) -> Path:
    if explicit is not None:
        wheel = Path(explicit)
        if not wheel.is_file():
            raise SystemExit(f"wheel not found: {wheel}")
        return wheel
    candidates = sorted(Path("dist").glob("slotvox-*.whl"), key=_version_key)
    if not candidates:
        raise SystemExit("no wheel found in dist/; run `make build` first")
    return candidates[-1]


def _run(command: list[str]) -> int:
    print("+", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-4000:], file=sys.stderr)
        print(result.stderr[-4000:], file=sys.stderr)
        return result.returncode
    if result.stdout.strip():
        print(result.stdout.strip()[:500])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install a built wheel into a fresh venv.")
    parser.add_argument("--wheel", default=None, help="wheel path (default: newest in dist/)")
    args = parser.parse_args(argv)
    wheel = find_wheel(args.wheel)

    with tempfile.TemporaryDirectory(prefix="slotvox-check-") as tmp:
        root = Path(tmp) / "venv"
        venv.create(root, with_pip=True)
        python = root / "bin" / "python"
        slotvox_bin = root / "bin" / "slotvox"
        steps = [
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                str(wheel),
            ],
            [str(python), "-c", IMPORT_CHECK],
            [str(slotvox_bin), "--help"],
        ]
        for command in steps:
            code = _run(command)
            if code != 0:
                return code
    print(f"OK: {wheel.name} installs, imports, and exposes a working CLI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
