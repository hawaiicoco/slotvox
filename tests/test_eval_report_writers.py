"""Report file writers: layout, contents, overwrite guard."""

import pytest

from slotvox.errors import ValidationError
from slotvox.eval.metrics import TurnRecord
from slotvox.eval.report import html_report, markdown_report, write_reports
from slotvox.eval.runs import ScoredRun


def build_run():
    records = (TurnRecord("a", "a", ("O",), ("O",), slices=(("noise", "clean"),)),)
    return ScoredRun.evaluate("writer-run", records)


def test_layout_and_contents(tmp_path):
    run = build_run()
    root = write_reports(run, tmp_path / "reports")
    markdown_path = root / "report.md"
    html_path = root / "report.html"
    assert markdown_path.is_file()
    assert html_path.is_file()
    assert markdown_path.read_text(encoding="utf-8") == markdown_report(run)
    assert html_path.read_text(encoding="utf-8") == html_report(run)


def test_overwrite_guard(tmp_path):
    run = build_run()
    root = write_reports(run, tmp_path / "reports")
    with pytest.raises(ValidationError, match="overwrite"):
        write_reports(run, root)
    write_reports(run, root, overwrite=True)


def test_run_type_validation(tmp_path):
    with pytest.raises(ValidationError, match="ScoredRun"):
        write_reports("nope", tmp_path / "x")
