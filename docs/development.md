# Development guide

## Environment

Python >= 3.11. The committed `uv.lock` reproduces the full dev environment
(including CPU-only torch):

```bash
uv sync --extra dev --extra torch
```

Without uv:

```bash
python -m pip install -e ".[dev]"
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Only CPU wheels are used. No GPU, pretrained weights, or external datasets
are needed for any test or example.

## Make targets

| Target | What it runs | Notes |
| ------ | ------------ | ----- |
| `build` | `python -m build --wheel --no-isolation` | wheel into `dist/` |
| `test` | `pytest -q -m "not slow"` | fast suite, seconds |
| `test-all` | `pytest -q` | includes `slow` + `model`; CI runs this |
| `format` | `ruff format` + `ruff check --fix` | run before committing |
| `format-check` | `ruff format --check` + `ruff check` | CI gate |
| `lint` | `ruff check` | CI gate |
| `typecheck` | `mypy` | advisory |
| `release` | build sdist+wheel, then `scripts/check_package.py` | installs the wheel into a throwaway venv and smoke-tests import + CLI |

## Test conventions

- Markers: `slow` (long training/integration; excluded from `make test`) and
  `model` (requires the torch extra; skipped cleanly without it).
- Tests cover boundaries and both directions of touched operations
  (encode/decode, serialize/deserialize, accept/reject).
- Public APIs, configuration schemas, and serialized outputs have
  golden/roundtrip tests; changing a golden must be a deliberate commit.
- Synthetic fixtures are labeled as synthetic in test names or docstrings.

## Commit and version conventions

- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`,
  `refactor:`) with short subjects.
- The version lives in `src/slotvox/_version.py`; during the 0.x series only
  patch bumps are made, and `CHANGELOG.md` is updated in the same commit.

## Repository layout

```text
src/slotvox/   package (see docs/architecture.md)
tests/         pytest suite mirroring the package layout
examples/      runnable offline examples, each with its own README
scripts/       packaging smoke test and helpers
docs/          documentation
```
