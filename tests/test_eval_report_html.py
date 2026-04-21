"""HTML report: escaping, self-containment, determinism."""

import pytest

from slotvox.errors import ValidationError
from slotvox.eval.metrics import TurnRecord
from slotvox.eval.report import html_report
from slotvox.eval.runs import ScoredRun


def hostile_run():
    return ScoredRun.evaluate(
        "run-hostile",
        (
            TurnRecord(
                "<img|x>",
                "<img|x>",
                (),
                (),
                slices=(("note", "<script>alert(1)</script>"),),
            ),
        ),
        metadata={"note": "<script>alert(1)</script>", 'q": ': "a&b"},
    )


def test_hostile_text_is_escaped():
    text = html_report(hostile_run())
    assert "<script" not in text
    assert "&lt;script&gt;" in text
    assert "&lt;img" in text
    assert "a&amp;b" in text


def test_self_contained_structure():
    text = html_report(hostile_run())
    assert text.startswith("<!DOCTYPE html>")
    assert "<style>" in text
    assert "src=" not in text
    assert "http" not in text
    for heading in (
        "<h2>Metadata</h2>",
        "<h2>Intent</h2>",
        "<h2>Joint turns</h2>",
        "<h2>Slices</h2>",
        "<h2>Limitations</h2>",
    ):
        assert heading in text
    assert "<li>" in text


def test_deterministic_and_typed():
    assert html_report(hostile_run()) == html_report(hostile_run())
    with pytest.raises(ValidationError, match="ScoredRun"):
        html_report(42)
