"""Markdown and self-contained HTML evaluation reports.

Every report carries a Limitations section — honest reporting is a
structural requirement, not an option. All dynamic text (ids, labels,
metadata values) is escaped, so hostile strings cannot inject markup:
Markdown cells escape pipes and newlines, HTML output escapes
everything and ships inline CSS only — no scripts, no external assets,
no network.
"""

from __future__ import annotations

import html
from pathlib import Path

from slotvox.errors import ValidationError
from slotvox.eval.runs import ScoredRun

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


def markdown_report(run: ScoredRun) -> str:
    """Render the full Markdown report for ``run``."""
    _require_run(run)
    lines = [f"# Evaluation report: {run.run_id}", "", "## Metadata", ""]
    if run.metadata:
        lines += ["| key | value |", "| --- | --- |"]
        for key in sorted(run.metadata):
            lines.append(f"| {_md_cell(key)} | {_md_cell(run.metadata[key])} |")
        lines.append("")
    else:
        lines += ["(none)", ""]
    lines += [f"Records: {run.record_count}", "", "## Intent", ""]
    lines += [
        f"Accuracy: {_fmt(run.intent.accuracy)} ({run.intent.correct}/{run.intent.total})",
        "",
        "## Slot F1 (both boundary policies)",
        "",
        "| policy | precision | recall | f1 | gold spans | pred spans |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, score in (("strict", run.slot_strict), ("partial", run.slot_partial)):
        lines.append(
            f"| {name} | {_fmt(score.precision)} | {_fmt(score.recall)} | {_fmt(score.f1)} "
            f"| {score.gold_count} | {score.pred_count} |"
        )
    lines += [
        "",
        "## Joint turns",
        "",
        f"Turn accuracy (strict): {_fmt(run.turn.accuracy)} ({run.turn.correct}/{run.turn.total})",
        "",
        "## Intent confusion (rows gold, columns predicted)",
        "",
    ]
    labels = run.confusion.labels
    lines.append("| gold \\ pred | " + " | ".join(_md_cell(label) for label in labels) + " |")
    lines.append("| --- |" * (len(labels) + 1))
    for row_index, label in enumerate(labels):
        cells = " | ".join(str(count) for count in run.confusion.counts[row_index])
        lines.append(f"| {_md_cell(label)} | {cells} |")
    lines += ["", "## Bootstrap CI (intent accuracy)", ""]
    if run.bootstrap is None:
        lines.append("Not computed for this run.")
    else:
        ci = run.bootstrap
        lines.append(
            f"Estimate {_fmt(ci.estimate)}, {ci.confidence:.0%} percentile CI "
            f"[{_fmt(ci.low)}, {_fmt(ci.high)}] over {ci.replicates} replicates "
            f"(seed {ci.seed})."
        )
    lines += ["", "## Slices", ""]
    if not run.slices:
        lines.append("No slice metadata on this run's records.")
    else:
        for kind in sorted(run.slices):
            lines += [
                f"### {kind}",
                "",
                "| value | count | intent acc | slot f1 (strict) | slot f1 (partial) |",
                "| --- | --- | --- | --- | --- |",
            ]
            for value in sorted(run.slices[kind]):
                entry = run.slices[kind][value]
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


_CSS = (
    "body{font-family:sans-serif;margin:2rem auto;max-width:60rem;color:#111}"
    "table{border-collapse:collapse;margin:0.5rem 0}"
    "th,td{border:1px solid #999;padding:0.25rem 0.5rem;text-align:left}"
    "th{background:#eee}"
)


def html_report(run: ScoredRun) -> str:
    """Render the self-contained HTML report (escaped, inline CSS, no scripts)."""
    _require_run(run)
    esc = html.escape
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>Evaluation report: {esc(run.run_id)}</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        f"<h1>Evaluation report: {esc(run.run_id)}</h1>",
        "<h2>Metadata</h2>",
    ]
    if run.metadata:
        parts.append("<table><tr><th>key</th><th>value</th></tr>")
        for key in sorted(run.metadata):
            parts.append(f"<tr><td>{esc(str(key))}</td><td>{esc(str(run.metadata[key]))}</td></tr>")
        parts.append("</table>")
    else:
        parts.append("<p>(none)</p>")
    parts.append(f"<p>Records: {run.record_count}</p>")
    parts.append("<h2>Intent</h2>")
    parts.append(
        f"<p>Accuracy: {_fmt(run.intent.accuracy)} ({run.intent.correct}/{run.intent.total})</p>"
    )
    parts.append("<h2>Slot F1 (both boundary policies)</h2>")
    parts.append(
        "<table><tr><th>policy</th><th>precision</th><th>recall</th><th>f1</th>"
        "<th>gold spans</th><th>pred spans</th></tr>"
    )
    for name, score in (("strict", run.slot_strict), ("partial", run.slot_partial)):
        parts.append(
            f"<tr><td>{name}</td><td>{_fmt(score.precision)}</td>"
            f"<td>{_fmt(score.recall)}</td><td>{_fmt(score.f1)}</td>"
            f"<td>{score.gold_count}</td><td>{score.pred_count}</td></tr>"
        )
    parts.append("</table>")
    parts.append("<h2>Joint turns</h2>")
    parts.append(
        f"<p>Turn accuracy (strict): {_fmt(run.turn.accuracy)} "
        f"({run.turn.correct}/{run.turn.total})</p>"
    )
    parts.append("<h2>Intent confusion (rows gold, columns predicted)</h2>")
    labels = run.confusion.labels
    head = "".join(f"<th>{esc(label)}</th>" for label in labels)
    parts.append(f"<table><tr><th>gold \\ pred</th>{head}</tr>")
    for row_index, label in enumerate(labels):
        cells = "".join(f"<td>{count}</td>" for count in run.confusion.counts[row_index])
        parts.append(f"<tr><td>{esc(label)}</td>{cells}</tr>")
    parts.append("</table>")
    parts.append("<h2>Bootstrap CI (intent accuracy)</h2>")
    if run.bootstrap is None:
        parts.append("<p>Not computed for this run.</p>")
    else:
        ci = run.bootstrap
        parts.append(
            f"<p>Estimate {_fmt(ci.estimate)}, {ci.confidence:.0%} percentile CI "
            f"[{_fmt(ci.low)}, {_fmt(ci.high)}] over {ci.replicates} replicates "
            f"(seed {ci.seed}).</p>"
        )
    parts.append("<h2>Slices</h2>")
    if not run.slices:
        parts.append("<p>No slice metadata on this run's records.</p>")
    else:
        for kind in sorted(run.slices):
            parts.append(f"<h3>{esc(kind)}</h3>")
            parts.append(
                "<table><tr><th>value</th><th>count</th><th>intent acc</th>"
                "<th>slot f1 (strict)</th><th>slot f1 (partial)</th></tr>"
            )
            for value in sorted(run.slices[kind]):
                entry = run.slices[kind][value]
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


def write_reports(run: ScoredRun, out_dir, *, overwrite: bool = False) -> Path:
    """Write ``report.md`` + ``report.html`` under ``out_dir``.

    The HTML file is written last and acts as the commit point; an
    existing report directory is an error unless ``overwrite`` is set.
    """
    _require_run(run)
    root = Path(out_dir)
    if (root / "report.html").exists() and not overwrite:
        raise ValidationError(f"{root} already holds reports; pass overwrite=True to replace")
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.md").write_text(markdown_report(run), encoding="utf-8")
    (root / "report.html").write_text(html_report(run), encoding="utf-8")
    return root
