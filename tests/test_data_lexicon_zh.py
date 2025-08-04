"""All zh lexicons satisfy their domain constraints (checked exhaustively)."""

from slotvox.data.lexicon import slot_entries
from slotvox.schema.builtins import builtin_domains
from slotvox.tagging.tokenize import tokenize

ZH_DOMAINS = ("calendar", "music-control", "navigation", "weather")


def test_every_domain_slot_has_zh_entries():
    for domain in builtin_domains():
        for slot in domain.slot_names:
            entries = slot_entries(domain.domain, "zh", slot)
            assert len(entries) >= 2, f"{domain.domain}/{slot} needs variety"


def test_all_zh_values_satisfy_constraints():
    for domain in builtin_domains():
        for slot in domain.slot_names:
            spec = domain.slot_spec(slot)
            for entry in slot_entries(domain.domain, "zh", slot):
                spec.check_value(entry.value)


def test_all_zh_surfaces_tokenize_cleanly():
    for domain in builtin_domains():
        for slot in domain.slot_names:
            for entry in slot_entries(domain.domain, "zh", slot):
                tokens = tokenize(entry.surface, "zh")
                assert len(tokens) >= 1
                assert "".join(tokens) == entry.surface


def test_zh_domains_covered():
    assert set(ZH_DOMAINS) == {domain.domain for domain in builtin_domains()}
