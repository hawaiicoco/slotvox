"""Golden pins for the public config API surface and cross-kind dispatch."""

from slotvox import config

GOLDEN_ALL = [
    "CONFIG_SCHEMA_VERSION",
    "Config",
    "ENCODERS",
    "FeatureConfig",
    "GenerationConfig",
    "SPLITS",
    "StreamConfig",
    "TrainConfig",
    "WINDOWS",
    "config_from_dict",
    "config_from_json",
    "config_to_json",
    "register_config",
    "registered_kinds",
]


def test_public_api_surface_is_pinned():
    assert sorted(config.__all__) == GOLDEN_ALL
    for name in GOLDEN_ALL:
        assert hasattr(config, name)


def test_registered_kinds_cover_all_config_records():
    assert {"feature", "generation", "train", "stream"} <= set(config.registered_kinds())


def test_dispatch_round_trips_every_kind():
    records = [
        config.FeatureConfig(),
        config.GenerationConfig(),
        config.TrainConfig(),
        config.StreamConfig(),
    ]
    for record in records:
        assert config.config_from_json(config.config_to_json(record)) == record
