"""Repository-level contracts: required docs exist and stay in sync.

These tests fail when documentation drifts from the package (version,
required Makefile targets, CI matrix), so drift must be deliberate.
"""

from pathlib import Path

import slotvox

REPO_ROOT = Path(__file__).resolve().parent.parent


def read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def test_readme_contains_timeline_statement():
    readme = read("README.md")
    assert "## 时间线说明" in readme
    assert "本项目于 2026 年 9 月创建和验证" in readme


def test_license_is_mit_and_names_holder():
    text = read("LICENSE")
    assert text.startswith("MIT License")
    assert "Zheng Haodi" in text


def test_changelog_tracks_package_version():
    assert f"## [{slotvox.__version__}]" in read("CHANGELOG.md")


def test_makefile_provides_required_targets():
    makefile = "\n" + read("Makefile")
    for target in (
        "build",
        "test",
        "test-all",
        "format",
        "format-check",
        "lint",
        "typecheck",
        "release",
    ):
        assert f"\n{target}:" in makefile, f"missing make target: {target}"


def test_ci_covers_supported_pythons_and_make_targets():
    ci = read(".github/workflows/ci.yml")
    for version in ("3.11", "3.12", "3.13"):
        assert f'"{version}"' in ci
    for target in ("make build", "make test-all", "make format-check", "make lint"):
        assert target in ci
