set dotenv-load := false

setup:
    uv sync --all-extras --dev

format:
    uv run ruff format src tests tools
    uv run ruff check --fix src tests tools

lint:
    uv run ruff check src tests tools

typecheck:
    uv run basedpyright

imports:
    uv run lint-imports --config .importlinter
    uv run python tools/check_public_imports.py

test:
    uv run pytest -m "not slow and not gpu and not pallas"

test-unit:
    uv run pytest -m "not slow and not gpu and not pallas"

test-integration:
    uv run pytest -m "not gpu and not pallas"

test-synthetic-smoke:
    uv run pytest -m "not slow and not gpu and not pallas"

check: format lint typecheck imports test

ci:
    uv run ruff format --check src tests tools
    uv run ruff check src tests tools
    uv run basedpyright
    uv run lint-imports --config .importlinter
    uv run python tools/check_public_imports.py
    uv run pytest -m "not slow and not gpu and not pallas"

bench-smoke:
    uv run python -m tomojax.datasets generate --manifest docs/tomojax-v2/benchmark_manifest.yaml --size 32 --out .artifacts/smoke
    uv run python -m tomojax.verify synthetic-smoke .artifacts/smoke

bench-128:
    uv run python -m tomojax.datasets generate --manifest docs/tomojax-v2/benchmark_manifest.yaml --size 128 --out .artifacts/bench128
    uv run python -m tomojax.benchmarks.run .artifacts/bench128
