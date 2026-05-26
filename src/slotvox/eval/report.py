"""Markdown and self-contained HTML evaluation reports.

Every report carries a Limitations section — honest reporting is a
structural requirement, not an option. Reports render from either a
live :class:`ScoredRun` or a serialized run envelope; both paths share
one view representation, so identical numbers produce identical output.
All dynamic text (ids, labels, metadata values) is escaped: Markdown
cells escape pipes and newlines, HTML escapes everything and ships
inline CSS only — no scripts, no external assets, no network.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from slotvox.errors import ValidationError
from slotvox.eval.runs import ScoredRun, validate_run_envelope

LIMITATIONS = (
    "All data is synthetic (slotvox factory signals); no real speech, "
    "real users, or pretrained models are involved.",
    "Decisions are greedy; no beam or CRF-style rescoring.",
    "Bootstrap intervals quantify sampling variability of this run on this dataset only.",
    "Numbers describe THIS run; they are not a claim about real-world quality.",
)


def _require_run(run) -> ScoredRun:
    if not isinstance(run, ScoredRun):
        raise ValidationError(f"run must be a ScoredRun, got {type(run).__name__}")
    return run


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _md_cell(value) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _score_view(score) -> dict[str, Any]:
    return {
        "precision": score.precision,
        "recall": score.recall,
        "f1": score.f1,
        "gold_count": score.gold_count,
        "pred_count": score.pred_count,
    }


def _run_view(run: ScoredRun) -> dict[str, Any]:
    """Plain-data view of a live run (shared by both renderers)."""
    return {
        "run_id": run.run_id,
        "metadata": run.metadata,
        "record_count": run.record_count,
        "intent": {
            "correct": run.intent.correct,
            "total": run.intent.total,
            "accuracy": run.intent.accuracy,
        },
        "slot_strict": _score_view(run.slot_strict),
        "slot_partial": _score_view(run.slot_partial),
        "turn": {
            "correct": run.turn.correct,
            "total": run.turn.total,
            "accuracy": run.turn.accuracy,
        },
        "labels": list(run.confusion.labels),
        "counts": [list(row) for row in run.confusion.counts],
        "bootstrap": None if run.bootstrap is None else run.bootstrap.to_dict(),
        "slices": run.slices,
    }


def _ratio(correct: Any, total: Any) -> float:
    return correct / total if total else 0.0


def _envelope_view(envelope: Any) -> dict[str, Any]:
    """Plain-data view of a serialized run envelope (strictly validated)."""
    validated = validate_run_envelope(envelope)
    intent = validated["intent"]
    turn = validated["turn"]
    return {
        "run_id": validated["run_id"],
        "metadata": validated["metadata"],
        "record_count": validated["record_count"],
        "intent": {
            "correct": intent["correct"],
            "total": intent["total"],
            "accuracy": _ratio(intent["correct"], intent["total"]),
        },
        "slot_strict": dict(validated["slot_strict"]),
        "slot_partial": dict(validated["slot_partial"]),
        "turn": {
            "correct": turn["correct"],
            "total": turn["total"],
            "accuracy": _ratio(turn["correct"], turn["total"]),
        },
        "labels": list(validated["confusion"]["labels"]),
        "counts": [list(row) for row in validated["confusion"]["counts"]],
        "bootstrap": validated["bootstrap"],
        "slices": validated["slices"],
    }


def _markdown_view(view: dict[str, Any]) -> str:
    lines = [f"# Evaluation report: {view['run_id']}", "", "## Metadata", ""]
    metadata = view["metadata"]
    if metadata:
        lines += ["| key | value |", "| --- | --- |"]
        for key in sorted(metadata):
            lines.append(f"| {_md_cell(key)} | {_md_cell(metadata[key])} |")
        lines.append("")
    else:
        lines += ["(none)", ""]
    intent = view["intent"]
    lines += [f"Records: {view['record_count']}", "", "## Intent", ""]
    lines += [
        f"Accuracy: {_fmt(intent['accuracy'])} ({intent['correct']}/{intent['total']})",
        "",
        "## Slot F1 (both boundary policies)",
        "",
        "| policy | precision | recall | f1 | gold spans | pred spans |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in ("slot_strict", "slot_partial"):
        score = view[name]
        policy = "strict" if name == "slot_strict" else "partial"
        lines.append(
            f"| {policy} | {_fmt(score['precision'])} | {_fmt(score['recall'])} "
            f"| {_fmt(score['f1'])} | {score['gold_count']} | {score['pred_count']} |"
        )
    turn = view["turn"]
    lines += [
        "",
        "## Joint turns",
        "",
        f"Turn accuracy (strict): {_fmt(turn['accuracy'])} ({turn['correct']}/{turn['total']})",
        "",
        "## Intent confusion (rows gold, columns predicted)",
        "",
    ]
    labels = view["labels"]
    lines.append("| gold \\ pred | " + " | ".join(_md_cell(label) for label in labels) + " |")
    lines.append("| --- |" * (len(labels) + 1))
    for row_index, label in enumerate(labels):
        cells = " | ".join(str(count) for count in view["counts"][row_index])
        lines.append(f"| {_md_cell(label)} | {cells} |")
    lines += ["", "## Bootstrap CI (intent accuracy)", ""]
    bootstrap = view["bootstrap"]
    if bootstrap is None:
        lines.append("Not computed for this run.")
    else:
        lines.append(
            f"Estimate {_fmt(bootstrap['estimate'])}, {bootstrap['confidence']:.0%} "
            f"percentile CI [{_fmt(bootstrap['low'])}, {_fmt(bootstrap['high'])}] over "
            f"{bootstrap['replicates']} replicates (seed {bootstrap['seed']})."
        )
    lines += ["", "## Slices", ""]
    slices = view["slices"]
    if not slices:
        lines.append("No slice metadata on this run's records.")
    else:
        for kind in sorted(slices):
            lines += [
                f"### {kind}",
                "",
                "| value | count | intent acc | slot f1 (strict) | slot f1 (partial) |",
                "| --- | --- | --- | --- | --- |",
            ]
            for value in sorted(slices[kind]):
                entry = slices[kind][value]
                lines.append(
                    f"| {_md_cell(value)} | {entry['count']} | "
                    f"{_fmt(entry['intent_accuracy'])} | {_fmt(entry['slot_f1_strict'])} | "
                    f"{_fmt(entry['slot_f1_partial'])} |"
                )
            lines.append("")
    lines += ["## Limitations", ""]
    lines += [f"- {item}" for item in LIMITATIONS]
    lines.append("")
    return "\n".join(lines)


def markdown_report(run: ScoredRun) -> str:
    """Render the full Markdown report for a live run."""
    return _markdown_view(_run_view(_require_run(run)))


def markdown_from_envelope(envelope: Any) -> str:
    """Render the Markdown report from a serialized run envelope."""
    return _markdown_view(_envelope_view(envelope))


_CSS = (
    "body{font-family:sans-serif;margin:2rem auto;max-width:60rem;color:#111}"
    "table{border-collapse:collapse;margin:0.5rem 0}"
    "th,td{border:1px solid #999;padding:0.25rem 0.5rem;text-align:left}"
    "th{background:#eee}"
)


def _html_view(view: dict[str, Any]) -> str:
    esc = html.escape
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>Evaluation report: {esc(view['run_id'])}</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        f"<h1>Evaluation report: {esc(view['run_id'])}</h1>",
        "<h2>Metadata</h2>",
    ]
    metadata = view["metadata"]
    if metadata:
        parts.append("<table><tr><th>key</th><th>value</th></tr>")
        for key in sorted(metadata):
            parts.append(f"<tr><td>{esc(str(key))}</td><td>{esc(str(metadata[key]))}</td></tr>")
        parts.append("</table>")
    else:
        parts.append("<p>(none)</p>")
    intent = view["intent"]
    parts.append(f"<p>Records: {view['record_count']}</p>")
    parts.append("<h2>Intent</h2>")
    parts.append(
        f"<p>Accuracy: {_fmt(intent['accuracy'])} ({intent['correct']}/{intent['total']})</p>"
    )
    parts.append("<h2>Slot F1 (both boundary policies)</h2>")
    parts.append(
        "<table><tr><th>policy</th><th>precision</th><th>recall</th><th>f1</th>"
        "<th>gold spans</th><th>pred spans</th></tr>"
    )
    for name in ("slot_strict", "slot_partial"):
        score = view[name]
        policy = "strict" if name == "slot_strict" else "partial"
        parts.append(
            f"<tr><td>{policy}</td><td>{_fmt(score['precision'])}</td>"
            f"<td>{_fmt(score['recall'])}</td><td>{_fmt(score['f1'])}</td>"
            f"<td>{score['gold_count']}</td><td>{score['pred_count']}</td></tr>"
        )
    parts.append("</table>")
    turn = view["turn"]
    parts.append("<h2>Joint turns</h2>")
    parts.append(
        f"<p>Turn accuracy (strict): {_fmt(turn['accuracy'])} "
        f"({turn['correct']}/{turn['total']})</p>"
    )
    parts.append("<h2>Intent confusion (rows gold, columns predicted)</h2>")
    labels = view["labels"]
    head = "".join(f"<th>{esc(label)}</th>" for label in labels)
    parts.append(f"<table><tr><th>gold \\ pred</th>{head}</tr>")
    for row_index, label in enumerate(labels):
        cells = "".join(f"<td>{count}</td>" for count in view["counts"][row_index])
        parts.append(f"<tr><td>{esc(label)}</td>{cells}</tr>")
    parts.append("</table>")
    parts.append("<h2>Bootstrap CI (intent accuracy)</h2>")
    bootstrap = view["bootstrap"]
    if bootstrap is None:
        parts.append("<p>Not computed for this run.</p>")
    else:
        parts.append(
            f"<p>Estimate {_fmt(bootstrap['estimate'])}, {bootstrap['confidence']:.0%} "
            f"percentile CI [{_fmt(bootstrap['low'])}, {_fmt(bootstrap['high'])}] over "
            f"{bootstrap['replicates']} replicates (seed {bootstrap['seed']}).</p>"
        )
    parts.append("<h2>Slices</h2>")
    slices = view["slices"]
    if not slices:
        parts.append("<p>No slice metadata on this run's records.</p>")
    else:
        for kind in sorted(slices):
            parts.append(f"<h3>{esc(kind)}</h3>")
            parts.append(
                "<table><tr><th>value</th><th>count</th><th>intent acc</th>"
                "<th>slot f1 (strict)</th><th>slot f1 (partial)</th></tr>"
            )
            for value in sorted(slices[kind]):
                entry = slices[kind][value]
                parts.append(
                    f"<tr><td>{esc(value)}</td><td>{entry['count']}</td>"
                    f"<td>{_fmt(entry['intent_accuracy'])}</td>"
                    f"<td>{_fmt(entry['slot_f1_strict'])}</td>"
                    f"<td>{_fmt(entry['slot_f1_partial'])}</td></tr>"
                )
            parts.append("</table>")
    parts.append("<h2>Limitations</h2>")
    parts.append("<ul>")
    for item in LIMITATIONS:
        parts.append(f"<li>{esc(item)}</li>")
    parts.append("</ul>")
    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts)


def html_report(run: ScoredRun) -> str:
    """Render the self-contained HTML report for a live run."""
    return _html_view(_run_view(_require_run(run)))


def html_from_envelope(envelope: Any) -> str:
    """Render the HTML report from a serialized run envelope."""
    return _html_view(_envelope_view(envelope))


def write_report_files(
    markdown_text: str, html_text: str, out_dir, *, overwrite: bool = False
) -> Path:
    """Write ``report.md`` + ``report.html`` under ``out_dir``.

    The HTML file is written last and acts as the commit point; an
    existing report directory is an error unless ``overwrite`` is set.
    """
    root = Path(out_dir)
    if (root / "report.html").exists() and not overwrite:
        raise ValidationError(f"{root} already holds reports; pass overwrite=True to replace")
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.md").write_text(markdown_text, encoding="utf-8")
    (root / "report.html").write_text(html_text, encoding="utf-8")
    return root


def write_reports(run: ScoredRun, out_dir, *, overwrite: bool = False) -> Path:
    """Render and write both report formats for a live run."""
    _require_run(run)
    return write_report_files(markdown_report(run), html_report(run), out_dir, overwrite=overwrite)
