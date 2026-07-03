"""The dataset-generation example runs end-to-end offline."""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "examples" / "dataset_generation" / "generate_and_validate.py"


def test_runs_and_validates(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    out = result.stdout
    assert "no leaks across train/dev/test" in out
    assert "dataset hash" in out
    assert "NOT real speech" in out
    assert (tmp_path / "dataset" / "dataset.json").is_file()


def test_english_language_flag(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(tmp_path), "--language", "en"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "first example:" in result.stdout
