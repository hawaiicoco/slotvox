"""`slotvox train` end-to-end on a tiny synthetic task (model extra)."""

import json

import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.model]

from slotvox.cli import main  # noqa: E402
from slotvox.models.checkpoint import load_checkpoint  # noqa: E402
from slotvox.schema.serialize import read_json  # noqa: E402

TINY = [
    "train",
    "--counts",
    "train=8,dev=4",
    "--speakers",
    "2",
    "--epochs",
    "1",
    "--hidden",
    "8",
    "--n-mels",
    "12",
    "--batch-size",
    "4",
    "--seed",
    "5",
]


def test_train_writes_checkpoint_history_and_spec(tmp_path, capsys):
    out = tmp_path / "model"
    assert main(TINY + ["--out", str(out), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert sorted(payload) == [
        "checkpoint",
        "dev_examples",
        "epochs",
        "final_dev_intent_accuracy",
        "final_dev_loss",
        "final_train_loss",
        "model_hash",
        "train_examples",
    ]
    assert payload["epochs"] == 1
    assert payload["train_examples"] == 8
    assert (out / "model.ckpt").is_file()
    history = read_json(out / "history.json")
    assert len(history) == 1
    assert history[0]["epoch"] == 0
    loaded = load_checkpoint(out / "model.ckpt")
    assert loaded.epoch == 1
    assert len(loaded.history) == 1
    assert loaded.spec.model_hash == payload["model_hash"]
    assert loaded.metrics["final_train_loss"] == history[0]["train_loss"]


def test_train_overwrite_guard(tmp_path, capsys):
    out = tmp_path / "model"
    assert main(TINY + ["--out", str(out)]) == 0
    capsys.readouterr()
    assert main(TINY + ["--out", str(out)]) == 2
    assert "overwrite" in capsys.readouterr().err
    assert main(TINY + ["--out", str(out), "--overwrite"]) == 0


def test_train_is_deterministic(tmp_path, capsys):
    first_out = tmp_path / "a"
    second_out = tmp_path / "b"
    assert main(TINY + ["--out", str(first_out), "--json"]) == 0
    first = json.loads(capsys.readouterr().out)["final_train_loss"]
    assert main(TINY + ["--out", str(second_out), "--json"]) == 0
    second = json.loads(capsys.readouterr().out)["final_train_loss"]
    assert first == second
