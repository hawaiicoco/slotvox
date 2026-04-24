"""`slotvox schema` output contracts."""

import json

from slotvox.cli import main
from slotvox.schema.builtins import BUILTIN_DOMAIN_IDS, builtin_domain
from slotvox.schema.domains import DomainSpec


def test_list_ids(capsys):
    assert main(["schema", "--list"]) == 0
    assert capsys.readouterr().out.split() == list(BUILTIN_DOMAIN_IDS)


def test_single_domain_json_roundtrip(capsys):
    assert main(["schema", "weather", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert sorted(payload) == ["domains"]
    assert len(payload["domains"]) == 1
    assert DomainSpec.from_dict(payload["domains"][0]) == builtin_domain("weather")


def test_all_domains_default(capsys):
    assert main(["schema", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["domains"]) == len(BUILTIN_DOMAIN_IDS)


def test_human_summary_and_unknown_domain(capsys):
    assert main(["schema", "weather"]) == 0
    out = capsys.readouterr().out
    assert "weather" in out
    assert "city" in out
    assert "query-weather" in out
    assert main(["schema", "nope"]) == 2
    assert "unknown built-in domain" in capsys.readouterr().err
