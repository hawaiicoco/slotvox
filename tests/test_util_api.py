"""Golden pin for the util package API surface."""

from slotvox import util
from slotvox.util.jsoncanon import stable_hash
from slotvox.util.seed import make_rng

GOLDEN_ALL = [
    "SEED_MAX",
    "canonical_dumps",
    "canonical_loads",
    "derive_seed",
    "make_rng",
    "stable_hash",
]


def test_util_api_surface_is_pinned():
    assert sorted(util.__all__) == GOLDEN_ALL
    for name in GOLDEN_ALL:
        assert hasattr(util, name)


def test_reexports_are_the_same_objects():
    assert util.stable_hash is stable_hash
    assert util.make_rng is make_rng
