"""tokenize/is_cjk contracts: languages, normalization, rejection."""

import pytest

from slotvox.errors import ValidationError
from slotvox.tagging.tokenize import is_cjk, tokenize


def test_zh_tokenizes_per_character():
    assert tokenize("北京明天", "zh") == ("北", "京", "明", "天")


def test_zh_drops_whitespace_keeps_punctuation():
    assert tokenize("北京 明天", "zh") == ("北", "京", "明", "天")
    assert tokenize("你好，世界", "zh") == ("你", "好", "，", "世", "界")


def test_en_tokenizes_per_word():
    assert tokenize("  play   some music ", "en") == ("play", "some", "music")


def test_nfc_normalization_unifies_equivalents():
    composed = tokenize("\u00e9clair", "en")
    decomposed = tokenize("e\u0301clair", "en")  # "e" + combining acute accent
    assert composed == decomposed == ("\u00e9clair",)
    assert len(composed[0]) == 6
    assert len("e\u0301clair") == 7  # NFC folds the combining accent into one char


def test_empty_and_invalid_inputs_rejected():
    with pytest.raises(ValidationError):
        tokenize("", "zh")
    with pytest.raises(ValidationError):
        tokenize("   ", "en")
    with pytest.raises(ValidationError):
        tokenize("x", "fr")
    with pytest.raises(ValidationError):
        tokenize(None, "zh")


def test_is_cjk_boundaries():
    assert is_cjk("中")
    assert is_cjk("，")
    assert is_cjk("Ａ")
    assert not is_cjk("a")
    assert not is_cjk("1")
    with pytest.raises(ValidationError):
        is_cjk("ab")
    with pytest.raises(ValidationError):
        is_cjk("")
