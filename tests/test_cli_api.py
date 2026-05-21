"""Golden pin for the CLI package surface."""

import re

from slotvox import cli
from slotvox._version import __version__
from slotvox.cli import main


def test_surface_pinned():
    assert cli.__all__ == ["main"]
    assert callable(main)


def test_version_is_patch_series():
    assert re.fullmatch(r"0\.\d+\.\d+", __version__)
