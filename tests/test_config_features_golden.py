"""Golden pins for FeatureConfig defaults; drift must be deliberate."""

from slotvox.config.base import config_from_json, config_to_json
from slotvox.config.features import FeatureConfig
from slotvox.util.jsoncanon import canonical_dumps

GOLDEN_DEFAULTS = {
    "kind": "feature",
    "schema_version": 1,
    "sample_rate": 16000,
    "frame_ms": 25,
    "hop_ms": 10,
    "n_fft": 512,
    "n_mels": 64,
    "fmin": 20.0,
    "fmax": 7600.0,
    "window": "hann",
}

GOLDEN_JSON = (
    '{"fmax":7600.0,"fmin":20.0,"frame_ms":25,"hop_ms":10,"kind":"feature",'
    '"n_fft":512,"n_mels":64,"sample_rate":16000,"schema_version":1,"window":"hann"}'
)


def test_default_to_dict_matches_golden():
    assert FeatureConfig().to_dict() == GOLDEN_DEFAULTS


def test_default_canonical_json_matches_golden_text():
    assert canonical_dumps(FeatureConfig().to_dict()) == GOLDEN_JSON


def test_defaults_survive_json_round_trip():
    assert config_from_json(config_to_json(FeatureConfig())) == FeatureConfig()
