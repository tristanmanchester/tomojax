set dotenv-load := false

benchmark_scripts := "bench"

# CPU tests run on one worker per two logical CPUs, each pinned to its share of the cores
# (tests/conftest.py) and handed one test at a time, the slow ones first.
parallel := "-n " + `echo $(( ($(nproc) + 1) / 2 ))` + " --maxschedchunk 1"

# GPU tests run on one worker per 1.3 GB of the first GPU's memory: a worker's CUDA
# context takes 0.3 GB and its JAX pool is capped at its share of the rest (tests/conftest.py).
gpu_parallel := "-n " + `(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo 0) | awk 'NR == 1 { print int($1 / 1300) }'` + " --maxschedchunk 1"

setup:
    uv sync --locked --extra cpu --dev

format:
    uv run --no-sync ruff format src tests tools examples {{benchmark_scripts}}
    uv run --no-sync ruff check --fix src tests tools examples {{benchmark_scripts}}

lint:
    uv run --no-sync ruff check src tests tools examples {{benchmark_scripts}}

docstrings:
    JAX_PLATFORMS=cpu uv run --no-sync python tools/check_docstrings.py

docs:
    JAX_PLATFORMS=cpu uv run --no-sync python tools/build_api_docs.py

typecheck:
    uv run --no-sync basedpyright

lock-check:
    uv lock --check

imports:
    uv run --no-sync lint-imports --config .importlinter
    uv run --no-sync python tools/check_public_imports.py

test:
    JAX_PLATFORMS=cpu uv run --no-sync pytest -q -m "not gpu" {{parallel}}

test-cov:
    JAX_PLATFORMS=cpu uv run --no-sync pytest -q tests -m "not gpu" --cov {{parallel}}

package:
    rm -rf dist
    uv build
    uv run --no-sync twine check --strict dist/*
    uv run --no-sync python tools/smoke_installed_wheel.py

# Rewrites the ratchet and API-surface records; tests/test_architecture.py is what checks them.
guardrails:
    JAX_PLATFORMS=cpu uv run --no-sync python tools/guardrails.py --write

smoke:
    uv run --no-sync python tools/smoke_cli_workflow.py

examples:
    uv run --no-sync python examples/simulate_and_reconstruct.py

accelerator-smoke:
    uv run --no-sync python tools/smoke_accelerator.py

accelerator-smoke-cuda:
    TOMOJAX_REQUIRE_CUDA=1 uv run --no-sync python tools/smoke_accelerator.py

# Every test with the GPU visible: unmarked tests take the CUDA paths too, which
# CI's CPU runner and `-m gpu` alone never exercise. The workers of the second stage
# share the GPU. About one run in three, XLA then fails to create a cuFFT plan in one
# worker and that worker's CUDA context is lost, failing every later test in it, so
# the tests that failed are run again in one process.
test-cuda: accelerator-smoke-cuda
    uv run --no-sync pytest -q -m fresh_gpu
    uv run --no-sync pytest -q -m "not fresh_gpu" {{gpu_parallel}} || uv run --no-sync pytest -q -m "not fresh_gpu" --last-failed --last-failed-no-failures none

benchmark-smoke:
    JAX_PLATFORMS=cpu uv run --no-sync python bench/compare_projectors.py --libraries tomojax --sizes 32 --views 30 --repeats 1
    JAX_PLATFORMS=cpu uv run --no-sync python bench/reconstruction_benchmark.py --sizes 32 --views 60 --repeats 1
    JAX_PLATFORMS=cpu uv run --no-sync python bench/workflow_benchmark.py --sizes 16 --geometries parallel --methods fista spdhg --repeats 1

check: format lint typecheck imports test

ci:
    uv lock --check
    uv run --no-sync ruff format --check src tests tools examples {{benchmark_scripts}}
    uv run --no-sync ruff check src tests tools examples {{benchmark_scripts}}
    uv run --no-sync basedpyright
    JAX_PLATFORMS=cpu uv run --no-sync python tools/check_docstrings.py
    uv run --no-sync lint-imports --config .importlinter
    uv run --no-sync python tools/check_public_imports.py
    rm -rf dist
    uv build
    uv run --no-sync twine check --strict dist/*
    uv run --no-sync python tools/smoke_installed_wheel.py
    uv run --no-sync python tools/smoke_cli_workflow.py
    uv run --no-sync python examples/simulate_and_reconstruct.py
    uv run --no-sync python tools/smoke_accelerator.py
    JAX_PLATFORMS=cpu uv run --no-sync pytest -q tests -m "not gpu and not slow" {{parallel}}
    just benchmark-smoke

surface-check:
    uv lock --check
    uv run --no-sync ruff format --check src tests tools examples {{benchmark_scripts}}
    uv run --no-sync ruff check src tests tools examples {{benchmark_scripts}}
    uv run --no-sync python tools/check_public_imports.py
    uv run --no-sync python tools/smoke_cli_workflow.py
    uv run --no-sync python tools/smoke_accelerator.py
    JAX_PLATFORMS=cpu uv run --no-sync pytest -q -m "surface and not numerical and not gpu" {{parallel}}

# Build the documentation site into site/out: the guides are copied from the
# repository's Markdown and the API pages generated from the source by griffe.
site:
    python3 tools/docs_sync.py
    uv run --no-project --with "griffe>=1.14" python tools/docs_api.py
    cd site && pnpm install --frozen-lockfile && pnpm build
