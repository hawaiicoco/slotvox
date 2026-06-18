"""`slotvox report` from run envelopes (torch-free)."""

import json

from slotvox.cli import main
from slotvox.eval.metrics import TurnRecord
from slotvox.eval.runs import ScoredRun
from slotvox.schema.serialize import write_json_atomic


def write_envelope(tmp_path):
    records = (
        TurnRecord("a", "a", ("B-city", "O"), ("B-city", "O"), slices=(("noise", "clean"),)),
        TurnRecord("a", "b", ("B-city", "O"), ("O", "O"), slices=(("noise", "snr-15"),)),
    )
    path = tmp_path / "run.json"
    write_json_atomic(path, ScoredRun.evaluate("report-cmd", records).to_dict())
    return path


def test_report_writes_both_formats(tmp_path, capsys):
    run_file = write_envelope(tmp_path)
    out = tmp_path / "rep"
    assert main(["report", "--run", str(run_file), "--out", str(out), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "report-cmd"
    assert payload["files"] == ["report.html", "report.md"]
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "# Evaluation report: report-cmd" in text
    assert (out / "report.html").is_file()


def test_report_bad_envelope_is_data_error(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    write_json_atomic(bad, {"nope": 1})
    assert main(["report", "--run", str(bad), "--out", str(tmp_path / "rep")]) == 3
    capsys.readouterr()
    code = main(["report", "--run", str(tmp_path / "missing.json"), "--out", str(tmp_path / "r")])
    assert code == 3
    assert "not found" in capsys.readouterr().err


def test_report_overwrite_guard(tmp_path, capsys):
    run_file = write_envelope(tmp_path)
    out = tmp_path / "rep"
    assert main(["report", "--run", str(run_file), "--out", str(out)]) == 0
    capsys.readouterr()
    assert main(["report", "--run", str(run_file), "--out", str(out)]) == 2
    assert "overwrite" in capsys.readouterr().err
