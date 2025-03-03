# Contributing to slotvox

Thanks for your interest! slotvox is a research toolkit for end-to-end spoken
language understanding experiments on fully synthetic speech.

## Ground rules

- Everything runs offline on synthetic data. Do not introduce real speech
  corpora, pretrained weights, network downloads, or GPU requirements into
  the default test/example path.
- NumPy is the only required runtime dependency. PyTorch belongs behind the
  optional `torch` extra; guard torch imports in tests with
  `pytest.importorskip("torch")` and mark such tests `model` (plus `slow`
  when they train).
- All randomness goes through explicit seeds; serialized artifacts carry
  schema versions and roundtrip/golden tests.
- Numbers in documentation must come from experiments actually executed in
  this repository, and reports keep their limitations sections honest.

## Development setup

```bash
uv sync --extra dev --extra torch
```

Then use the Makefile: `make build`, `make test`, `make test-all`,
`make format-check`, `make lint`, `make typecheck`. See
[docs/development.md](docs/development.md) for details.

## Pull request checklist

- [ ] `make build`, `make test-all`, `make format-check`, `make lint` pass
- [ ] New behavior ships with focused tests (boundaries + both directions)
- [ ] Serialized formats have golden/roundtrip tests
- [ ] Docs updated; no quality claims beyond executed synthetic experiments
- [ ] Conventional commit messages; patch-level version changes only

## Code style

`ruff format` + `ruff check` (line length 100), dataclasses and type hints
throughout, `mypy` for static checks. Prefer small, single-purpose modules.
