"""`slotvox generate` end-to-end (in-process, offline)."""

import json

from slotvox.cli import main
from slotvox.data.persist import load_dataset


def test_generate_writes_loadable_dataset(tmp_path, capsys):
    out = tmp_path / "ds"
    code = main(
        [
            "generate",
            "--domain",
            "weather",
            "--counts",
            "train=4,dev=2",
            "--speakers",
            "2",
            "--seed",
            "7",
            "--out",
            str(out),
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert sorted(payload) == [
        "counts",
        "dataset_hash",
        "domain",
        "language",
        "path",
        "policy",
    ]
    assert payload["counts"] == {"train": 4, "dev": 2, "test": 0}
    assert load_dataset(out).dataset_hash == payload["dataset_hash"]


def test_generate_is_deterministic(tmp_path, capsys):
    args = ["generate", "--counts", "train=3", "--speakers", "1", "--seed", "9"]
    assert main(args + ["--out", str(tmp_path / "a"), "--json"]) == 0
    first = json.loads(capsys.readouterr().out)["dataset_hash"]
    assert main(args + ["--out", str(tmp_path / "b"), "--json"]) == 0
    second = json.loads(capsys.readouterr().out)["dataset_hash"]
    assert first == second


def test_human_output_shape(tmp_path, capsys):
    assert main(["generate", "--counts", "train=2", "--speakers", "1", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "generated 2 examples (train=2)" in out
    assert "dataset hash:" in out


def test_overwrite_guard_and_bad_counts(tmp_path, capsys):
    out = tmp_path / "ds"
    base = ["generate", "--counts", "train=2", "--speakers", "1"]
    assert main(base + ["--out", str(out)]) == 0
    capsys.readouterr()
    assert main(base + ["--out", str(out)]) == 2
    assert "already holds" in capsys.readouterr().err
    assert main(base + ["--out", str(tmp_path / "x"), "--counts", "train=x"]) == 2
    assert "must be an int" in capsys.readouterr().err
