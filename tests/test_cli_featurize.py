"""`slotvox featurize` end-to-end (in-process, offline)."""

import json

from slotvox.cli import main


def make_dataset(tmp_path, capsys):
    dataset_dir = tmp_path / "ds"
    assert (
        main(["generate", "--counts", "train=2", "--speakers", "1", "--out", str(dataset_dir)]) == 0
    )
    capsys.readouterr()
    return dataset_dir


def test_featurize_pipeline(tmp_path, capsys):
    dataset_dir = make_dataset(tmp_path, capsys)
    feat = tmp_path / "feat"
    assert (
        main(
            [
                "featurize",
                "--dataset",
                str(dataset_dir),
                "--out",
                str(feat),
                "--n-mels",
                "8",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert sorted(payload) == ["dataset_hash", "example_count", "n_mels", "path", "sample_rate"]
    assert payload["example_count"] == 2
    assert payload["n_mels"] == 8
    assert (feat / "features.json").is_file()


def test_missing_dataset_is_data_error(tmp_path, capsys):
    code = main(["featurize", "--dataset", str(tmp_path / "nope"), "--out", str(tmp_path / "f")])
    assert code == 3
    assert "not found" in capsys.readouterr().err


def test_bad_feature_config_is_usage_error(tmp_path, capsys):
    dataset_dir = make_dataset(tmp_path, capsys)
    code = main(
        [
            "featurize",
            "--dataset",
            str(dataset_dir),
            "--out",
            str(tmp_path / "f"),
            "--n-mels",
            "0",
        ]
    )
    assert code == 2
    assert "error" in capsys.readouterr().err
