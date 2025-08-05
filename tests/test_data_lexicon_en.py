"""All en lexicons satisfy their domain constraints (checked exhaustively)."""

from slotvox.data.lexicon import registered_lexicons, slot_entries
from slotvox.schema.builtins import builtin_domains
from slotvox.tagging.tokenize import tokenize


def test_en_lexicons_registered_for_all_domains():
    keys = set(registered_lexicons())
    for domain in builtin_domains():
        assert (domain.domain, "en") in keys
        assert (domain.domain, "zh") in keys


def test_all_en_values_satisfy_constraints():
    for domain in builtin_domains():
        for slot in domain.slot_names:
            spec = domain.slot_spec(slot)
            entries = slot_entries(domain.domain, "en", slot)
            assert len(entries) >= 2
            for entry in entries:
                spec.check_value(entry.value)


def test_en_surfaces_tokenize_into_words():
    for domain in builtin_domains():
        for slot in domain.slot_names:
            for entry in slot_entries(domain.domain, "en", slot):
                tokens = tokenize(entry.surface, "en")
                assert len(tokens) >= 1
                assert " ".join(tokens) == entry.surface
