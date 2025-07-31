"""Golden pin for the tagging package API surface."""

from slotvox import tagging

GOLDEN_ALL = [
    "BIOES_PREFIXES",
    "BIO_PREFIXES",
    "OUTSIDE",
    "REPAIR_POLICIES",
    "Span",
    "bio_to_bioes",
    "bioes_to_bio",
    "format_tag",
    "is_cjk",
    "is_valid_sequence",
    "is_valid_tag",
    "parse_tag",
    "repair_sequence",
    "spans_to_tags",
    "tags_to_spans",
    "tokenize",
    "validate_sequence",
]


def test_tagging_api_surface_is_pinned():
    assert sorted(tagging.__all__) == GOLDEN_ALL
    for name in GOLDEN_ALL:
        assert hasattr(tagging, name)


def test_text_and_tag_alignment_workflow():
    tokens = tagging.tokenize("播放 青花瓷", "en")  # mixed text, en mode splits words
    assert tokens == ("播放", "青花瓷")
    tokens_zh = tagging.tokenize("播放青花瓷", "zh")
    tags = ("B-song", "I-song", "I-song", "I-song", "I-song")
    assert len(tags) == len(tokens_zh)
    spans = tagging.tags_to_spans(tags)
    assert spans == (tagging.Span("song", 0, 5),)
    span = spans[0]
    assert "".join(tokens_zh[span.start : span.end]) == "播放青花瓷"
