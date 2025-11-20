"""InstructionSample structural rules: alternation, system, audio."""

import pytest

from slotvox.errors import ValidationError
from slotvox.instructions.schema import AudioRef, InstructionSample, Turn

HASH = "c" * 64


def audio_turn():
    return Turn("user", audio=AudioRef("weather-train-00000", HASH, 500.0))


def sample(turns=None, **overrides):
    data = {
        "sample_id": "weather-train-00000",
        "domain": "weather",
        "language": "zh",
        "intent": "query-weather",
        "turns": turns
        if turns is not None
        else (
            Turn("system", text="prompt"),
            audio_turn(),
            Turn("assistant", text="意图：query-weather"),
        ),
        "tags": ("synthetic",),
    }
    data.update(overrides)
    return InstructionSample(**data)


def test_valid_shapes():
    built = sample()
    assert len(built.turns) == 3
    assert built.turns[1].audio is not None
    without_system = sample(
        turns=(
            audio_turn(),
            Turn("assistant", text="a"),
            audio_turn(),
            Turn("assistant", text="b"),
        )
    )
    assert len(without_system.turns) == 4


def test_system_rules():
    with pytest.raises(ValidationError, match="at most one system"):
        sample(
            turns=(
                Turn("system", text="p1"),
                Turn("system", text="p2"),
                audio_turn(),
                Turn("assistant", text="a"),
            )
        )


def test_alternation_rules():
    with pytest.raises(ValidationError, match="must be 'user'"):
        sample(
            turns=(
                Turn("system", text="p"),
                Turn("assistant", text="a"),
                audio_turn(),
            )
        )
    with pytest.raises(ValidationError, match="pairs"):
        sample(turns=(audio_turn(), Turn("assistant", text="a"), audio_turn()))
    with pytest.raises(ValidationError, match="pairs"):
        sample(turns=(audio_turn(),))


def test_audio_turn_required():
    with pytest.raises(ValidationError, match="audio turn"):
        sample(
            turns=(
                Turn("user", text="上海今天天气怎么样"),
                Turn("assistant", text="a"),
            )
        )


def test_field_validation():
    with pytest.raises(ValidationError, match="language"):
        sample(language="fr")
    with pytest.raises(ValidationError):
        sample(sample_id="Bad Id")
    with pytest.raises(ValidationError, match="unique"):
        sample(tags=("synthetic", "synthetic"))
    with pytest.raises(ValidationError):
        sample(tags=("Bad Tag",))
    with pytest.raises(ValidationError, match="Turn"):
        sample(turns=("nope", audio_turn()))
