.PHONY: help install lock test lint format check clean

POETRY ?= poetry
PYTHON ?= python

help:
	@$(POETRY) --version
	@echo "Targets:"
	@echo "  install  Install project and development dependencies"
	@echo "  lock     Resolve and update poetry.lock"
	@echo "  test     Run the test suite"
	@echo "  lint     Run Ruff checks"
	@echo "  format   Format source and tests with Ruff"
	@echo "  check    Run lint and tests"
	@echo "  clean    Remove generated Python caches and build artifacts"

install:
	$(POETRY) install

lock:
	$(POETRY) lock

test:
	$(POETRY) run pytest

lint:
	$(POETRY) run ruff check src tests

format:
	$(POETRY) run ruff format src tests

check: lint test

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(pathlib.Path(p), ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', '.mypy_cache', 'htmlcov', 'dist', 'build', 'cover']]"
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]"
