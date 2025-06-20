"""SlotTypeSpec serialization contracts, including a golden dict."""

import pytest

from slotvox.errors import ValidationError
from slotvox.schema.slots import SlotTypeSpec
from slotvox.util.jsoncanon import stable_hash

SPECS = [
    SlotTypeSpec("city", "enum", {"choices": ["北京", "上海"]}, "目标城市"),
    SlotTypeSpec("temperature", "number", {"minimum": -40.0, "maximum": 60.0, "unit": "°C"}),
    SlotTypeSpec("depart", "time", description="出发时间"),
    SlotTypeSpec("note", "text", {"min_length": 2, "max_length": 64}),
]


@pytest.mark.parametrize("index", range(len(SPECS)))
def test_round_trip_every_type(index):
    spec = SPECS[index]
    assert SlotTypeSpec.from_dict(spec.to_dict()) == spec


def test_to_dict_is_json_native():
    for spec in SPECS:
        payload = spec.to_dict()
        assert isinstance(stable_hash(payload), str)
        if spec.value_type == "enum":
            assert isinstance(payload["constraint"]["choices"], list)


def test_from_dict_strict_keys():
    payload = SPECS[0].to_dict()
    with pytest.raises(ValidationError, match="unknown keys"):
        SlotTypeSpec.from_dict({**payload, "extra": 1})
    missing = dict(payload)
    del missing["value_type"]
    with pytest.raises(ValidationError, match="missing keys"):
        SlotTypeSpec.from_dict(missing)
    with pytest.raises(ValidationError, match="must be a dict"):
        SlotTypeSpec.from_dict([payload])


def test_golden_enum_spec_dict():
    assert SPECS[0].to_dict() == {
        "name": "city",
        "value_type": "enum",
        "constraint": {"choices": ["北京", "上海"]},
        "description": "目标城市",
    }
