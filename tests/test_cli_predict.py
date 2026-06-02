"""`slotvox predict` end-to-end (model extra, synthetic audio)."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.model]

from slotvox.audio import WavAudio, write_wav  # noqa: E402
from slotvox.cli import main  # noqa: E402
from slotvox.config import GenerationConfig  # noqa: E402
from slotvox.data.dataset import generate_dataset  # noqa: E402

TINY = [
    "train",
    "--counts",
    "train=6,dev=2",
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
    "13",
]


def train_tiny(tmp_path):
    out = tmp_path / "model"
    assert main(TINY + ["--out", str(out)]) == 0
    return out / "model.ckpt"


def make_wav(tmp_path, name="clip-one.wav", sample_rate=16000):
    dataset = generate_dataset(
        GenerationConfig(domain="weather", seed=17, counts={"train": 1}, n_speakers=1)
    )
    samples = np.asarray(dataset.examples[0].utterance.samples, dtype=np.float32)
    path = tmp_path / name
    write_wav(path, WavAudio(sample_rate=sample_rate, samples=samples))
    return path


def test_predict_roundtrip(tmp_path, capsys):
    checkpoint = train_tiny(tmp_path)
    capsys.readouterr()
    wav = make_wav(tmp_path)
    assert main(["predict", "--model", str(checkpoint), "--audio", str(wav), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["intent"] in ("query-weather", "query-forecast", "weather-alert")
    assert 0.0 <= payload["posterior"] <= 1.0
    assert payload["frame_count"] > 0
    assert isinstance(payload["spans"], list)
    for span in payload["spans"]:
        assert sorted(span) == ["end_frame", "label", "start_frame"]


def test_predict_human_output(tmp_path, capsys):
    checkpoint = train_tiny(tmp_path)
    capsys.readouterr()
    wav = make_wav(tmp_path)
    assert main(["predict", "--model", str(checkpoint), "--audio", str(wav)]) == 0
    out = capsys.readouterr().out
    assert "intent:" in out and "frames:" in out


def test_predict_sample_rate_mismatch(tmp_path, capsys):
    checkpoint = train_tiny(tmp_path)
    capsys.readouterr()
    wav = make_wav(tmp_path, name="narrow.wav", sample_rate=8000)
    code = main(["predict", "--model", str(checkpoint), "--audio", str(wav)])
    assert code == 2
    assert "sample rate" in capsys.readouterr().err


def test_predict_missing_files_are_data_errors(tmp_path, capsys):
    wav = make_wav(tmp_path)
    assert main(["predict", "--model", str(tmp_path / "nope.ckpt"), "--audio", str(wav)]) == 3
    capsys.readouterr()
    checkpoint = train_tiny(tmp_path)
    capsys.readouterr()
    code = main(["predict", "--model", str(checkpoint), "--audio", str(tmp_path / "nope.wav")])
    assert code == 3
    assert "not found" in capsys.readouterr().err
