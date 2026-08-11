"""The joint-training example shows real movement (slow, model extra)."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.slow, pytest.mark.model]

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "examples" / "joint_training" / "train_joint_model.py"
RESULT = r"RESULT (\w+) loss=([\d.]+) intent_acc=([\d.]+) frame_acc=([\d.]+)"


def test_training_improves_dev_metrics(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(tmp_path), "--epochs", "6"],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    matches = dict(
        (phase, (float(loss), float(intent), float(frame)))
        for phase, loss, intent, frame in re.findall(RESULT, result.stdout)
    )
    assert set(matches) == {"before", "after"}
    before, after = matches["before"], matches["after"]
    assert after[0] < before[0], "dev loss must decrease"
    assert after[1] > before[1], "dev intent accuracy must improve"
    assert after[2] > before[2], "dev frame accuracy must improve"
    assert (tmp_path / "model.ckpt").is_file()
    assert "not a benchmark" in result.stdout
