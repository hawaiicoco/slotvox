"""Turn contracts: roles, audio channel rules, and serde."""

import pytest

from slotvox.errors import ValidationError
from slotvox.instructions.schema import AudioRef, Turn

HASH = "b" * 64


def ref():
    return AudioRef("weather-train-00000", HASH, 500.0)


def test_text_and_audio_turns():
    system = Turn("system", text="prompt")
    assert system.audio is None
    user = Turn("user", audio=ref())
    assert user.text is None
    assert user.audio is not None
    both = Turn("user", text="transcript", audio=ref())
    assert both.text == "transcript"


def test_audio_restricted_to_user_turns():
    with pytest.raises(ValidationError, match="user turns"):
        Turn("assistant", audio=ref())
    with pytest.raises(ValidationError, match="user turns"):
        Turn("system", audio=ref())


def test_channel_and_role_validation():
    with pytest.raises(ValidationError, match="role"):
        Turn("tool", text="x")
    with pytest.raises(ValidationError, match="text, audio"):
        Turn("user")
    with pytest.raises(ValidationError, match="non-blank"):
        Turn("system", text="   ")
    with pytest.raises(ValidationError, match="AudioRef"):
        Turn("user", audio="not-a-ref")


def test_roundtrip_and_strict_keys():
    turn = Turn("user", audio=ref())
    payload = turn.to_dict()
    assert sorted(payload) == ["audio", "role", "text"]
    assert Turn.from_dict(payload) == turn
    with pytest.raises(ValidationError, match="unknown keys"):
        Turn.from_dict({**payload, "extra": 1})
    with pytest.raises(ValidationError, match="missing keys"):
        Turn.from_dict({"role": "user"})
