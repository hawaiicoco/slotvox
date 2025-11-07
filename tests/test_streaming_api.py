"""Golden pin for the streaming package API surface."""

import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.model]

import slotvox.streaming as streaming  # noqa: E402
from slotvox.streaming.session import StreamSession  # noqa: E402

GOLDEN_ALL = ["AudioBuffer", "IntentHypothesis", "StreamResult", "StreamSession"]


def test_streaming_api_surface_is_pinned():
    assert sorted(streaming.__all__) == GOLDEN_ALL
    for name in GOLDEN_ALL:
        assert hasattr(streaming, name)


def test_lazy_names_resolve_to_real_objects():
    assert streaming.StreamSession is StreamSession
    assert streaming.AudioBuffer(4).capacity == 4


def test_unknown_attribute_rejected():
    with pytest.raises(AttributeError):
        streaming.nonexistent_name
