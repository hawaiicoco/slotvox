"""Markdown report contents, escaping, and determinism."""

import pytest

from slotvox.errors import ValidationError
from slotvox.eval.metrics import TurnRecord
from slotvox.eval.report import markdown_report
from slotvox.eval.runs import ScoredRun


def build_run(**kwargs):
    records = (
        TurnRecord("a", "a", ("B-city", "O"), ("B-city", "O"), slices=(("noise", "clean"),)),
        TurnRecord("a", "b", ("B-city", "O"), ("O", "O"), slices=(("noise", "snr-15"),)),
        TurnRecord("b", "b", ("O",), ("O",), slices=(("noise", "snr-15"),)),
    )
    return ScoredRun.evaluate("report-run", records, metadata={"model": "synthetic-tcn"}, **kwargs)


def test_sections_and_numbers():
    text = markdown_report(build_run())
    assert text.startswith("# Evaluation report: report-run")
    for heading in (
        "## Metadata",
        "## Intent",
        "## Slot F1 (both boundary policies)",
        "## Joint turns",
        "## Intent confusion (rows gold, columns predicted)",
        "## Bootstrap CI (intent accuracy)",
        "## Slices",
        "## Limitations",
    ):
        assert heading in text
    assert "| strict |" in text
    assert "| partial |" in text
    assert "0.6667 (2/3)" in text
    assert "synthetic-tcn" in text
    assert "Not computed for this run." in text
    assert "synthetic" in text  # the limitations say so honestly


def test_bootstrap_section_when_computed():
    text = markdown_report(build_run(bootstrap_seed=5, replicates=50))
    assert "percentile CI" in text
    assert "50 replicates" in text
    assert "Not computed" not in text


def test_hostile_cells_escaped():
    run = ScoredRun.evaluate("r", (TurnRecord("a|b", "a|b", (), ()),), metadata={"note": "x|y\nz"})
    text = markdown_report(run)
    assert "a\\|b" in text
    assert "| a|b |" not in text
    assert "x\\|y z" in text


def test_deterministic_and_typed():
    assert markdown_report(build_run()) == markdown_report(build_run())
    with pytest.raises(ValidationError, match="ScoredRun"):
        markdown_report("nope")
