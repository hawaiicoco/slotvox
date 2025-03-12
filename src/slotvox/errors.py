"""Exception hierarchy for slotvox.

Every error raised by the library derives from :class:`SlotvoxError`, so
callers can catch library failures without swallowing unrelated exceptions.
Data-validation failures additionally derive from :class:`ValidationError`.
"""


class SlotvoxError(Exception):
    """Base class for all errors raised by slotvox."""


class ValidationError(SlotvoxError):
    """Data violates a schema, contract, or value constraint."""


class SchemaError(ValidationError):
    """A serialized artifact does not match its declared schema or version."""


class AudioError(ValidationError):
    """Audio bytes or parameters violate the WAV/PCM contract."""


class ConfigError(ValidationError):
    """A configuration object, field, or file is invalid."""


class TaggingError(ValidationError):
    """A tag sequence violates the BIO/BIOES tag algebra."""


class AdapterError(SlotvoxError):
    """An inference adapter violates the request/response protocol."""


class StreamingError(SlotvoxError):
    """A streaming session is driven outside its state machine."""
