"""The instruction-export example runs end-to-end (torch-free)."""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "examples" / "instruction_export" / "export_and_evaluate.py"


def test_runs_and_writes_reports(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    out = result.stdout
    assert "intent accuracy: 1.0000 (gold replay)" in out
    assert "BY CONSTRUCTION" in out
    assert (tmp_path / "corpus" / "instructions.jsonl").is_file()
    assert (tmp_path / "corpus" / "manifest.json").is_file()
    assert (tmp_path / "fixtures.json").is_file()
    assert (tmp_path / "run.json").is_file()
    assert (tmp_path / "report" / "report.md").is_file()
    assert (tmp_path / "report" / "report.html").is_file()
