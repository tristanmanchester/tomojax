set dotenv-load := false

benchmark_scripts := "bench"

setup:
    uv sync --locked --extra cpu --dev

format:
    uv run --no-sync ruff format src tests tools examples {{benchmark_scripts}}
    uv run --no-sync ruff check --fix src tests tools examples {{benchmark_scripts}}

lint:
    uv run --no-sync ruff check src tests tools examples {{benchmark_scripts}}

typecheck:
    uv run --no-sync basedpyright

lock-check:
    uv lock --check

imports:
    uv run --no-sync lint-imports --config .importlinter
    uv run --no-sync python tools/check_public_imports.py

test:
    JAX_PLATFORMS=cpu uv run --no-sync pytest -q -m "not gpu"

test-cov:
    JAX_PLATFORMS=cpu uv run --no-sync pytest -q tests -m "not gpu" --cov

package:
    rm -rf dist
    uv build
    uv run --no-sync twine check dist/*
    uv run --no-sync python tools/smoke_installed_wheel.py

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

test-cuda: accelerator-smoke-cuda
    uv run --no-sync pytest -q -m gpu

benchmark-smoke:
    JAX_PLATFORMS=cpu uv run --no-sync python bench/compare_projectors.py --libraries tomojax --sizes 32 --views 30 --repeats 1
    JAX_PLATFORMS=cpu uv run --no-sync python bench/reconstruction_benchmark.py --sizes 32 --views 60 --repeats 1
    JAX_PLATFORMS=cpu uv run --no-sync python bench/adjoint_benchmark.py --sizes 16 --repeats 1
    JAX_PLATFORMS=cpu uv run --no-sync python bench/adjoint_benchmark.py --stack --sizes 16 --batches 2 --repeats 1
    JAX_PLATFORMS=cpu uv run --no-sync python bench/workflow_benchmark.py --sizes 16 --geometries parallel --methods fista spdhg --repeats 1

check: format lint typecheck imports test

ci:
    uv lock --check
    uv run --no-sync ruff format --check src tests tools examples {{benchmark_scripts}}
    uv run --no-sync ruff check src tests tools examples {{benchmark_scripts}}
    uv run --no-sync basedpyright
    uv run --no-sync lint-imports --config .importlinter
    uv run --no-sync python tools/check_public_imports.py
    rm -rf dist
    uv build
    uv run --no-sync twine check dist/*
    uv run --no-sync python tools/smoke_installed_wheel.py
    uv run --no-sync python tools/smoke_cli_workflow.py
    uv run --no-sync python examples/simulate_and_reconstruct.py
    uv run --no-sync python tools/smoke_accelerator.py
    JAX_PLATFORMS=cpu uv run --no-sync pytest -q tests -m "not gpu" --cov
    just benchmark-smoke

surface-check:
    uv lock --check
    uv run --no-sync ruff format --check src tests tools examples {{benchmark_scripts}}
    uv run --no-sync ruff check src tests tools examples {{benchmark_scripts}}
    uv run --no-sync python tools/check_public_imports.py
    uv run --no-sync python tools/smoke_cli_workflow.py
    uv run --no-sync python tools/smoke_accelerator.py
    JAX_PLATFORMS=cpu uv run --no-sync pytest -q -m "surface and not numerical and not gpu"
