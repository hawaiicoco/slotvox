"""encode_split contracts: guards, vocab strictness, alignment checks."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.model]

from slotvox.config import FeatureConfig, GenerationConfig  # noqa: E402
from slotvox.data.dataset import generate_dataset  # noqa: E402
from slotvox.data.features_store import featurize_dataset  # noqa: E402
from slotvox.data.persist import save_dataset  # noqa: E402
from slotvox.errors import SchemaError, ValidationError  # noqa: E402
from slotvox.models.dataset import encode_split  # noqa: E402
from slotvox.models.vocab import Vocab, build_intent_vocab, build_tag_vocab  # noqa: E402
from slotvox.schema.builtins import weather_domain  # noqa: E402
from slotvox.schema.serialize import read_jsonl, write_jsonl  # noqa: E402

DOMAIN = weather_domain()
INTENTS = Vocab(build_intent_vocab(DOMAIN))
TAGS = Vocab(build_tag_vocab(DOMAIN))
FEATURES = FeatureConfig(n_mels=12)


def stored(tmp_path, seed=41, name="ds"):
    config = GenerationConfig(
        domain="weather",
        seed=seed,
        counts={"train": 4, "dev": 2},
        n_speakers=2,
        noise_snr_db=(40.0,),
    )
    dataset = generate_dataset(config)
    root = save_dataset(dataset, tmp_path / name)
    features = featurize_dataset(dataset, FEATURES, tmp_path / f"{name}-feat")
    return dataset, root, features


def test_encode_split_counts_and_contents(tmp_path):
    dataset, root, features = stored(tmp_path)
    train = encode_split(root, features, "train", intents=INTENTS, tags=TAGS)
    dev = encode_split(root, features, "dev", intents=INTENTS, tags=TAGS)
    assert (len(train), len(dev)) == (4, 2)
    for example in train:
        assert example.mel.shape[0] == example.frame_tag_indices.shape[0]
        assert len(example.frame_tag_names) == example.n_frames
        assert 0 <= example.intent_index < len(INTENTS)
        assert example.mel.dtype == np.float32


def test_feature_store_from_other_dataset_rejected(tmp_path):
    _, root_a, _ = stored(tmp_path, seed=41, name="a")
    _, _, features_b = stored(tmp_path, seed=42, name="b")
    with pytest.raises(SchemaError, match="does not belong"):
        encode_split(root_a, features_b, "train", intents=INTENTS, tags=TAGS)


def test_unknown_intent_label_rejected(tmp_path):
    _, root, features = stored(tmp_path)
    rows = read_jsonl(root / "manifest.jsonl")
    rows[0]["annotation"]["intent"] = "play-music"
    write_jsonl(root / "manifest.jsonl", rows)
    with pytest.raises(SchemaError, match="unknown vocab entry"):
        encode_split(root, features, "train", intents=INTENTS, tags=TAGS)


def test_alignment_disagreement_rejected(tmp_path):
    dataset, root, features = stored(tmp_path)
    victim = dataset.examples[0].utterance_id
    with np.load(features / f"{victim}.npz") as bundle:
        mel = bundle["mel"]
        token_map = np.zeros_like(bundle["frame_token"])
    np.savez(features / f"{victim}.npz", mel=mel, frame_token=token_map)
    with pytest.raises(SchemaError, match="disagrees with the manifest"):
        encode_split(root, features, "train", intents=INTENTS, tags=TAGS)


def test_bad_split_and_vocab_rejected(tmp_path):
    _, root, features = stored(tmp_path)
    with pytest.raises(ValidationError, match="split must be"):
        encode_split(root, features, "nope", intents=INTENTS, tags=TAGS)
    with pytest.raises(ValidationError, match="Vocab"):
        encode_split(root, features, "train", intents="x", tags=TAGS)
