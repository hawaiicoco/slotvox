"""Envelope-rendered reports: parity with live-run rendering."""

import pytest

from slotvox.errors import SchemaError, ValidationError
from slotvox.eval.metrics import TurnRecord
from slotvox.eval.report import (
    html_from_envelope,
    html_report,
    markdown_from_envelope,
    markdown_report,
    write_report_files,
)
from slotvox.eval.runs import ScoredRun


def build_run():
    records = (
        TurnRecord("a", "a", ("B-city", "O"), ("B-city", "O"), slices=(("noise", "clean"),)),
        TurnRecord("a", "b", ("B-city", "O"), ("O", "O"), slices=(("noise", "snr-15"),)),
        TurnRecord("b", "b", ("O",), ("O",), slices=(("noise", "snr-15"),)),
    )
    return ScoredRun.evaluate(
        "parity-run", records, metadata={"model": "synthetic"}, bootstrap_seed=5, replicates=50
    )


def test_markdown_parity():
    run = build_run()
    assert markdown_from_envelope(run.to_dict()) == markdown_report(run)


def test_html_parity():
    run = build_run()
    assert html_from_envelope(run.to_dict()) == html_report(run)


def test_envelope_validation_applies():
    run = build_run()
    envelope = run.to_dict()
    with pytest.raises(SchemaError, match="schema"):
        markdown_from_envelope({**envelope, "schema": "other"})
    with pytest.raises(ValidationError, match="dict|envelope"):
        markdown_from_envelope("nope")


def test_write_report_files_guard(tmp_path):
    root = write_report_files("# md", "<html></html>", tmp_path / "rep")
    assert (root / "report.md").read_text(encoding="utf-8") == "# md"
    with pytest.raises(ValidationError, match="overwrite"):
        write_report_files("# md", "<html></html>", root)
    write_report_files("# md2", "<html></html>", root, overwrite=True)
    assert (root / "report.md").read_text(encoding="utf-8") == "# md2"
