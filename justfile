set dotenv-load := false

setup:
    uv sync --extra cpu --dev

format:
    uv run ruff format src tests tools examples
    uv run ruff check --fix src tests tools examples

lint:
    uv run ruff check src tests tools examples

typecheck:
    uv run basedpyright

imports:
    uv run lint-imports --config .importlinter
    uv run python tools/check_public_imports.py

test:
    uv run pytest -q

test-cov:
    uv run pytest -q tests --cov

check: format lint typecheck imports test

ci:
    uv run ruff format --check src tests tools examples
    uv run ruff check src tests tools examples
    uv run basedpyright
    uv run lint-imports --config .importlinter
    uv run python tools/check_public_imports.py
    uv run pytest -q tests --cov

surface-check:
    uv run ruff format --check src tests tools examples
    uv run ruff check src tests tools examples
    uv run python tools/check_public_imports.py
    uv run pytest -q tests --cov
