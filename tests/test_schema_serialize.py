"""JSON/JSONL file helper contracts: atomicity, bounds, line errors."""

import pytest

from slotvox.errors import SchemaError
from slotvox.schema.serialize import (
    iter_jsonl,
    read_json,
    read_jsonl,
    write_json_atomic,
    write_jsonl,
)


def test_json_round_trip_and_no_tmp_leftovers(tmp_path):
    target = tmp_path / "nested" / "obj.json"
    write_json_atomic(target, {"b": 1, "a": [1, 2]})
    assert read_json(target) == {"b": 1, "a": [1, 2]}
    assert not list(tmp_path.rglob("*.tmp"))


def test_read_json_rejects_missing_oversized_and_broken(tmp_path):
    with pytest.raises(SchemaError, match="not found"):
        read_json(tmp_path / "absent.json")
    target = tmp_path / "big.json"
    target.write_text('{"a":1}', encoding="utf-8")
    with pytest.raises(SchemaError, match="exceeds bound"):
        read_json(target, max_bytes=4)
    with pytest.raises(SchemaError, match="max_bytes"):
        read_json(target, max_bytes=0)
    broken = tmp_path / "broken.json"
    broken.write_bytes(b'{"a":"\xff\xfe"}')
    with pytest.raises(SchemaError, match="utf-8"):
        read_json(broken)


def test_jsonl_round_trip(tmp_path):
    target = tmp_path / "rows.jsonl"
    rows = [{"i": 0, "text": "北京"}, {"i": 1, "text": "上海"}]
    write_jsonl(target, rows)
    assert read_jsonl(target) == rows


def test_jsonl_line_errors_carry_numbers(tmp_path):
    target = tmp_path / "rows.jsonl"
    target.write_text('{"a":1}\n{oops}\n', encoding="utf-8")
    with pytest.raises(SchemaError, match="line 2"):
        read_jsonl(target)
    target.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
    with pytest.raises(SchemaError, match="line 2: empty"):
        read_jsonl(target)
    target.write_text("[1,2]\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="expected a JSON object"):
        read_jsonl(target)


def test_write_jsonl_reports_bad_rows(tmp_path):
    with pytest.raises(SchemaError, match="row 1"):
        write_jsonl(tmp_path / "rows.jsonl", [{"a": 1}, {"bad": object()}])


def test_iter_jsonl_is_lazy(tmp_path):
    target = tmp_path / "rows.jsonl"
    target.write_text('{"a":1}\n{oops}\n', encoding="utf-8")
    iterator = iter_jsonl(target)
    assert next(iterator) == {"a": 1}
    with pytest.raises(SchemaError, match="line 2"):
        next(iterator)
