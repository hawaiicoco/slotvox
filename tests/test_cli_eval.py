"""`slotvox eval` end-to-end on a tiny trained model (model extra)."""

import json

import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.model]

from slotvox.cli import main  # noqa: E402
from slotvox.eval.runs import validate_run_envelope  # noqa: E402
from slotvox.schema.serialize import read_json  # noqa: E402


def setup_model_and_dataset(tmp_path, capsys, domain="weather"):
    dataset_dir = tmp_path / "ds"
    code = main(
        [
            "generate",
            "--domain",
            domain,
            "--counts",
            "train=4,dev=2,test=2",
            "--speakers",
            "3",
            "--out",
            str(dataset_dir),
        ]
    )
    assert code == 0
    capsys.readouterr()
    model_dir = tmp_path / "model"
    code = main(
        [
            "train",
            "--domain",
            domain,
            "--counts",
            "train=4,dev=2",
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
            "21",
            "--out",
            str(model_dir),
        ]
    )
    assert code == 0
    capsys.readouterr()
    return dataset_dir, model_dir / "model.ckpt"


def test_eval_writes_valid_envelope(tmp_path, capsys):
    dataset_dir, checkpoint = setup_model_and_dataset(tmp_path, capsys)
    run_file = tmp_path / "run.json"
    code = main(
        [
            "eval",
            "--dataset",
            str(dataset_dir),
            "--model",
            str(checkpoint),
            "--split",
            "test",
            "--run-id",
            "cli-eval-test",
            "--bootstrap-seed",
            "5",
            "--replicates",
            "50",
            "--out",
            str(run_file),
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["records"] == 2
    assert 0.0 <= payload["intent_accuracy"] <= 1.0
    assert 0.0 <= payload["slot_f1_strict"] <= 1.0
    envelope = read_json(run_file)
    assert validate_run_envelope(envelope) is envelope
    assert envelope["run_id"] == "cli-eval-test"
    assert envelope["metadata"]["split"] == "test"
    assert envelope["bootstrap"] is not None


def test_eval_human_output_is_honest(tmp_path, capsys):
    dataset_dir, checkpoint = setup_model_and_dataset(tmp_path, capsys)
    assert main(["eval", "--dataset", str(dataset_dir), "--model", str(checkpoint)]) == 0
    out = capsys.readouterr().out
    assert "synthetic data" in out
    assert "intent accuracy:" in out


def test_eval_empty_split_is_usage_error(tmp_path, capsys):
    _, checkpoint = setup_model_and_dataset(tmp_path, capsys)
    bare = tmp_path / "bare"
    assert main(["generate", "--counts", "train=2", "--speakers", "1", "--out", str(bare)]) == 0
    capsys.readouterr()
    code = main(["eval", "--dataset", str(bare), "--model", str(checkpoint), "--split", "test"])
    assert code == 2
    assert "empty" in capsys.readouterr().err


def test_eval_domain_mismatch_rejected(tmp_path, capsys):
    _, checkpoint = setup_model_and_dataset(tmp_path, capsys, domain="music-control")
    weather_dir = tmp_path / "weather-ds"
    code = main(
        [
            "generate",
            "--domain",
            "weather",
            "--counts",
            "train=2",
            "--speakers",
            "1",
            "--out",
            str(weather_dir),
        ]
    )
    assert code == 0
    capsys.readouterr()
    code = main(["eval", "--dataset", str(weather_dir), "--model", str(checkpoint)])
    assert code == 2
    assert "does not match" in capsys.readouterr().err
