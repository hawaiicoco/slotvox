"""load_features/list_features contracts: recompute, reject, guard paths."""

import numpy as np
import pytest

from slotvox.config import FeatureConfig, GenerationConfig
from slotvox.data.alignment import frame_token_map
from slotvox.data.dataset import generate_dataset
from slotvox.data.features_store import featurize_dataset, list_features, load_features
from slotvox.errors import SchemaError
from slotvox.features.frontend import log_mel


def stored(tmp_path, n_mels=8):
    dataset = generate_dataset(
        GenerationConfig(domain="weather", seed=33, counts={"train": 2}, n_speakers=1)
    )
    config = FeatureConfig(n_mels=n_mels)
    root = featurize_dataset(dataset, config, tmp_path / "feat")
    return dataset, config, root


def test_round_trip_matches_recomputation(tmp_path):
    dataset, config, root = stored(tmp_path)
    assert list_features(root) == tuple(sorted(e.utterance_id for e in dataset.examples))
    example = dataset.examples[0]
    mel, token_map = load_features(root, example.utterance_id)
    assert np.array_equal(mel, log_mel(example.utterance.samples, config))
    assert np.array_equal(token_map, frame_token_map(example.utterance, config))
    assert mel.dtype == np.float32
    assert token_map.dtype == np.int64


def test_unknown_id_and_bad_ids_rejected(tmp_path):
    _, _, root = stored(tmp_path)
    with pytest.raises(SchemaError, match="no feature array"):
        load_features(root, "missing-00000")
    for bad in ("", "has space", "../escape", "a/b", "a\\b", ".."):
        with pytest.raises(SchemaError, match="invalid utterance id"):
            load_features(root, bad)


def test_foreign_arrays_rejected(tmp_path):
    dataset, _, root = stored(tmp_path)
    victim = dataset.examples[0].utterance_id
    np.savez(root / f"{victim}.npz", junk=np.zeros(3))
    with pytest.raises(SchemaError, match="unexpected arrays"):
        load_features(root, victim)


def test_count_mismatch_rejected(tmp_path):
    _, _, root = stored(tmp_path)
    next(root.glob("*.npz")).unlink()
    with pytest.raises(SchemaError, match="envelope says"):
        list_features(root)


def test_missing_envelope_rejected(tmp_path):
    with pytest.raises(SchemaError, match="not found"):
        list_features(tmp_path / "nowhere")
