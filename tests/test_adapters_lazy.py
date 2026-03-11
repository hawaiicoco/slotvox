"""Lazy torch export resolution for the adapters package."""

import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.model]

import slotvox.adapters as adapters  # noqa: E402
from slotvox.adapters.model import JointAdapter  # noqa: E402


def test_lazy_export_resolves():
    assert adapters.JointAdapter is JointAdapter


def test_unknown_attribute_rejected():
    with pytest.raises(AttributeError):
        adapters.nonexistent_name
