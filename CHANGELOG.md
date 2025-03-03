# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog; versioning uses semantic versioning (0.x series).

## [0.1.0]

### Added

- Package scaffold: src layout, hatchling build, NumPy-only runtime core,
  optional CPU-only `torch` extra, pinned `uv.lock`.
- Developer tooling: pytest with `slow`/`model` markers, ruff format+lint,
  mypy, and a wheel install smoke test (`scripts/check_package.py`).
- Project documentation: README (with timeline note), architecture overview,
  development guide, references, contributing and security guides.
- CI matrix (ubuntu-latest, Python 3.11/3.12/3.13) mirroring the Makefile,
  tag-triggered GitHub release workflow, issue/PR templates, dependabot.
