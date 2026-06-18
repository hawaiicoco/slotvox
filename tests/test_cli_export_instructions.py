"""`slotvox export-instructions` end-to-end (torch-free)."""

import json

from slotvox.cli import main
from slotvox.instructions.export import load_instructions
from slotvox.schema.serialize import read_json


def make_dataset(tmp_path, capsys):
    dataset_dir = tmp_path / "ds"
    code = main(["generate", "--counts", "train=3", "--speakers", "1", "--out", str(dataset_dir)])
    assert code == 0
    capsys.readouterr()
    return dataset_dir


def test_export_corpus(tmp_path, capsys):
    dataset_dir = make_dataset(tmp_path, capsys)
    out = tmp_path / "corpus"
    code = main(["export-instructions", "--dataset", str(dataset_dir), "--out", str(out), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert sorted(payload) == ["content_hash", "path", "quality", "sample_count"]
    assert payload["sample_count"] == 3
    assert payload["quality"]["passed"] == 3
    assert len(load_instructions(out)) == 3
    assert read_json(out / "manifest.json")["content_hash"] == payload["content_hash"]


def test_export_with_snr_threshold(tmp_path, capsys):
    dataset_dir = make_dataset(tmp_path, capsys)
    out = tmp_path / "corpus-snr"
    code = main(
        [
            "export-instructions",
            "--dataset",
            str(dataset_dir),
            "--out",
            str(out),
            "--min-snr-db",
            "20",
            "--json",
        ]
    )
    assert code == 0
    quality = json.loads(capsys.readouterr().out)["quality"]
    assert quality["passed"] + quality["failed"] == 3


def test_export_guards(tmp_path, capsys):
    dataset_dir = make_dataset(tmp_path, capsys)
    out = tmp_path / "corpus"
    assert main(["export-instructions", "--dataset", str(dataset_dir), "--out", str(out)]) == 0
    capsys.readouterr()
    assert main(["export-instructions", "--dataset", str(dataset_dir), "--out", str(out)]) == 2
    assert "overwrite" in capsys.readouterr().err
    code = main(
        ["export-instructions", "--dataset", str(tmp_path / "nope"), "--out", str(tmp_path / "x")]
    )
    assert code == 3
