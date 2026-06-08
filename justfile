set dotenv-load := false

setup:
    uv sync --locked --extra cpu --dev

format:
    uv run ruff format src tests tools examples
    uv run ruff check --fix src tests tools examples

lint:
    uv run ruff check src tests tools examples

typecheck:
    uv run basedpyright

lock-check:
    uv lock --check

imports:
    uv run lint-imports --config .importlinter
    uv run python tools/check_public_imports.py

test:
    uv run pytest -q

test-cov:
    uv run pytest -q tests --cov

package:
    rm -rf dist
    uv build
    uv run twine check dist/*
    uv run python tools/smoke_installed_wheel.py

smoke:
    uv run python tools/smoke_cli_workflow.py

accelerator-smoke:
    uv run python tools/smoke_accelerator.py

accelerator-smoke-cuda:
    TOMOJAX_REQUIRE_CUDA=1 uv run python tools/smoke_accelerator.py

check: format lint typecheck imports test

ci:
    uv lock --check
    uv run ruff format --check src tests tools examples
    uv run ruff check src tests tools examples
    uv run basedpyright
    uv run lint-imports --config .importlinter
    uv run python tools/check_public_imports.py
    rm -rf dist
    uv build
    uv run twine check dist/*
    uv run python tools/smoke_installed_wheel.py
    uv run python tools/smoke_cli_workflow.py
    uv run python tools/smoke_accelerator.py
    uv run pytest -q tests --cov

surface-check:
    uv lock --check
    uv run ruff format --check src tests tools examples
    uv run ruff check src tests tools examples
    uv run python tools/check_public_imports.py
    uv run python tools/smoke_cli_workflow.py
    uv run python tools/smoke_accelerator.py
    uv run pytest -q tests --cov
