"""StreamConfig validation and roundtrip contracts."""

import pytest

from slotvox.config.streaming import StreamConfig
from slotvox.errors import ConfigError


def test_defaults_round_trip_and_chunk_geometry():
    config = StreamConfig()
    assert StreamConfig.from_dict(config.to_dict()) == config
    assert config.chunk_samples == 2560


def test_buffer_capacity_must_cover_four_chunks():
    with pytest.raises(ConfigError, match="buffer_capacity_ms"):
        StreamConfig(chunk_ms=160, buffer_capacity_ms=320)
    assert StreamConfig(chunk_ms=160, buffer_capacity_ms=640).buffer_capacity_ms == 640


def test_lookahead_bounded_by_capacity():
    with pytest.raises(ConfigError, match="lookahead_ms"):
        StreamConfig(buffer_capacity_ms=640, lookahead_ms=800)
    assert StreamConfig(lookahead_ms=0).lookahead_ms == 0


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
def test_stability_threshold_within_open_unit_interval(threshold):
    with pytest.raises(ConfigError, match=r"\(0, 1\]"):
        StreamConfig(stability_threshold=threshold)


def test_stability_threshold_boundary_accepted():
    assert StreamConfig(stability_threshold=1.0).stability_threshold == 1.0
