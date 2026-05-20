"""`--counts` parsing: strict in both directions."""

import pytest

from slotvox.cli.dataset import parse_counts
from slotvox.errors import ValidationError


def test_parses_pairs():
    assert parse_counts("train=8,dev=2") == {"train": 8, "dev": 2}
    assert parse_counts("test=0") == {"test": 0}
    assert parse_counts("train=1,dev=2,test=3") == {"train": 1, "dev": 2, "test": 3}


def test_rejections():
    for bad in ("", "8", "nope=1", "train=x", "train=-1", "train=1,train=2", "train=1,,dev=2"):
        with pytest.raises(ValidationError):
            parse_counts(bad)
    with pytest.raises(ValidationError, match="unknown split"):
        parse_counts("valid=1")
    with pytest.raises(ValidationError, match="duplicate"):
        parse_counts("dev=1,dev=2")
