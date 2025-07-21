"""Bounded JSON/JSONL file helpers for schema artifacts.

Writes are atomic (temp file + rename), so readers never observe partial
artifacts. Reads enforce an explicit byte cap before parsing, and JSONL
parse errors carry the offending line number.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from slotvox.errors import SchemaError
from slotvox.util.jsoncanon import canonical_dumps, canonical_loads

MAX_ARTIFACT_BYTES = 256 * 1024 * 1024


def _check_cap(max_bytes: int) -> None:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise SchemaError(f"max_bytes must be an int, got {max_bytes!r}")
    if max_bytes <= 0 or max_bytes > MAX_ARTIFACT_BYTES:
        raise SchemaError(f"max_bytes must be within (0, {MAX_ARTIFACT_BYTES}], got {max_bytes}")


def _write_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def write_json_atomic(path: str | Path, obj: Any) -> Path:
    """Serialize ``obj`` as canonical JSON and write it atomically."""
    text = canonical_dumps(obj)
    return _write_atomic(Path(path), text + "\n")


def read_json(path: str | Path, *, max_bytes: int = MAX_ARTIFACT_BYTES) -> Any:
    """Read and parse a canonical JSON file under a size bound."""
    _check_cap(max_bytes)
    source = Path(path)
    if not source.is_file():
        raise SchemaError(f"JSON file not found: {source}")
    size = source.stat().st_size
    if size > max_bytes:
        raise SchemaError(f"JSON file {source} is {size} bytes, exceeds bound {max_bytes}")
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SchemaError(f"JSON file is not valid utf-8: {source}") from exc
    return canonical_loads(text)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write rows as canonical JSONL (one compact object per line)."""
    lines: list[str] = []
    for index, row in enumerate(rows):
        try:
            lines.append(canonical_dumps(row))
        except SchemaError as exc:
            raise SchemaError(f"row {index}: {exc}") from exc
    text = "".join(line + "\n" for line in lines)
    return _write_atomic(Path(path), text)


def iter_jsonl(
    path: str | Path, *, max_bytes: int = MAX_ARTIFACT_BYTES
) -> Iterator[dict[str, Any]]:
    """Iterate JSONL rows, raising SchemaError with line numbers on problems."""
    _check_cap(max_bytes)
    source = Path(path)
    if not source.is_file():
        raise SchemaError(f"JSONL file not found: {source}")
    size = source.stat().st_size
    if size > max_bytes:
        raise SchemaError(f"JSONL file {source} is {size} bytes, exceeds bound {max_bytes}")
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SchemaError(f"JSONL file is not valid utf-8: {source}") from exc
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise SchemaError(f"line {lineno}: empty line in JSONL file {source}")
        try:
            obj = canonical_loads(line)
        except SchemaError as exc:
            raise SchemaError(f"line {lineno}: {exc}") from exc
        if not isinstance(obj, dict):
            raise SchemaError(f"line {lineno}: expected a JSON object, got {type(obj).__name__}")
        yield obj


def read_jsonl(path: str | Path, *, max_bytes: int = MAX_ARTIFACT_BYTES) -> list[dict[str, Any]]:
    """Read all JSONL rows into a list."""
    return list(iter_jsonl(path, max_bytes=max_bytes))
