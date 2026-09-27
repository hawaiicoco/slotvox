#!/usr/bin/env python3
"""Smoke-test built slotvox distributions in throwaway virtual environments.

Catches packaging mistakes that in-tree tests cannot see: missing files,
broken console-script entry points, or undeclared runtime dependencies.

Usage:
    make release                       # build sdist + wheel, then this check
    python scripts/check_package.py    # check the newest wheel in dist/
    python scripts/check_package.py --wheel dist/slotvox-0.1.0-py3-none-any.whl
    python scripts/check_package.py --all-distributions
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


def _version_key(artifact: Path) -> tuple[int, ...]:
    match = re.match(r"slotvox-(\d+)\.(\d+)\.(\d+)", artifact.name)
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


def find_distributions() -> tuple[Path, Path]:
    """Return the newest matching wheel and source distribution."""
    wheel = find_wheel(None)
    sdists = sorted(Path("dist").glob("slotvox-*.tar.gz"), key=_version_key)
    if not sdists:
        raise SystemExit("no sdist found in dist/; run `python -m build` first")
    sdist = sdists[-1]
    if _version_key(wheel) != _version_key(sdist):
        raise SystemExit(f"wheel and sdist versions differ: {wheel.name} vs {sdist.name}")
    return wheel, sdist


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


def _check_artifact(artifact: Path) -> int:
    """Install one artifact in isolation and exercise its public entry points."""
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
                str(artifact),
            ],
            [str(python), "-c", IMPORT_CHECK],
            [str(slotvox_bin), "--help"],
        ]
        for command in steps:
            code = _run(command)
            if code != 0:
                return code
    print(f"OK: {artifact.name} installs, imports, and exposes a working CLI")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install built distributions into fresh virtual environments."
    )
    parser.add_argument("--wheel", default=None, help="wheel path (default: newest in dist/)")
    parser.add_argument(
        "--all-distributions",
        action="store_true",
        help="check the newest matching wheel and source distribution",
    )
    args = parser.parse_args(argv)
    if args.wheel is not None and args.all_distributions:
        parser.error("--wheel and --all-distributions are mutually exclusive")
    artifacts = find_distributions() if args.all_distributions else (find_wheel(args.wheel),)
    for artifact in artifacts:
        code = _check_artifact(artifact)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
