"""Bootstrap smoke tests: the package imports and exposes a sane version."""

import re
from importlib import metadata

import slotvox


def test_version_is_plain_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", slotvox.__version__)


def test_installed_metadata_matches_package_version():
    assert metadata.version("slotvox") == slotvox.__version__
