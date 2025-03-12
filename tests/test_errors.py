"""Contracts for the slotvox exception hierarchy."""

import pytest

from slotvox import errors


def exception_classes():
    return [
        getattr(errors, name)
        for name in dir(errors)
        if isinstance(getattr(errors, name), type)
        and issubclass(getattr(errors, name), BaseException)
    ]


def test_all_errors_derive_from_slotvox_error():
    classes = exception_classes()
    assert len(classes) >= 8
    for cls in classes:
        assert issubclass(cls, errors.SlotvoxError)


def test_validation_family_derives_from_validation_error():
    family = (errors.SchemaError, errors.AudioError, errors.ConfigError, errors.TaggingError)
    for cls in family:
        assert issubclass(cls, errors.ValidationError)


def test_adapter_and_streaming_errors_are_not_validation_errors():
    assert not issubclass(errors.AdapterError, errors.ValidationError)
    assert not issubclass(errors.StreamingError, errors.ValidationError)


@pytest.mark.parametrize(
    "cls", [errors.SlotvoxError, errors.SchemaError, errors.AdapterError, errors.StreamingError]
)
def test_messages_round_trip(cls):
    with pytest.raises(cls, match="boom-42"):
        raise cls("boom-42")
