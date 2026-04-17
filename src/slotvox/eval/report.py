"""Markdown and self-contained HTML evaluation reports.

Every report carries a Limitations section — honest reporting is a
structural requirement, not an option. All dynamic text (ids, labels,
metadata values) is escaped, so hostile strings cannot inject markup:
Markdown cells escape pipes and newlines, HTML output escapes
everything and ships inline CSS only — no scripts, no external assets,
no network.
"""

from __future__ import annotations

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
