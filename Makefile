PY := .venv/bin/python
RUFF := .venv/bin/ruff
SRC_DIRS := src tests examples scripts

.PHONY: build test test-all format format-check lint typecheck release clean

build:
	$(PY) -m build --wheel --no-isolation

test:
	$(PY) -m pytest -q -m "not slow"

test-all:
	$(PY) -m pytest -q

format:
	$(RUFF) format $(SRC_DIRS)
	$(RUFF) check --fix $(SRC_DIRS)

format-check:
	$(RUFF) format --check $(SRC_DIRS)
	$(RUFF) check $(SRC_DIRS)

lint:
	$(RUFF) check $(SRC_DIRS)

typecheck:
	$(PY) -m mypy

release:
	rm -rf dist
	$(PY) -m build --no-isolation
	$(PY) scripts/check_package.py

clean:
	rm -rf dist build .pytest_cache .ruff_cache .mypy_cache
	find . -name '__pycache__' -type d -not -path './.venv/*' -exec rm -rf {} +
