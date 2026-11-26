set dotenv-load := false

# Keep temporary builds and test files off /tmp unless the caller chose a location.
export TMPDIR := if env("TMPDIR", "") == "" {
    `mkdir -p -- "$HOME/.cache/tomojax-tmp" && printf '%s' "$HOME/.cache/tomojax-tmp"`
} else {
    env("TMPDIR")
}

benchmark_scripts := "bench"
studio_target_dir := `printf '%s' "${CARGO_TARGET_DIR:-$HOME/.cache/tomojax-studio-target}"`
studio_toolchain := "1.99.1"

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

# Built in this checkout: the shared target directory holds whichever worktree's binary was built last.
studio-wheel:
    rm -rf dist-studio
    cargo build --locked --release --manifest-path studio/Cargo.toml --features static-hdf5 --target-dir studio/target
    uv run --no-sync python tools/build_studio_wheel.py --out-dir dist-studio --check-static
    uv run --no-sync twine check --strict dist-studio/*.whl

[env("CARGO_TARGET_DIR", studio_target_dir)]
[env("RUSTUP_TOOLCHAIN", studio_toolchain)]
studio-check: studio-lint
    cargo test --locked --manifest-path studio/Cargo.toml

[private]
[env("CARGO_TARGET_DIR", studio_target_dir)]
[env("RUSTUP_TOOLCHAIN", studio_toolchain)]
studio-lint:
    cargo fmt --manifest-path studio/Cargo.toml --check
    cargo clippy --locked --manifest-path studio/Cargo.toml --all-targets -- -D warnings

# Rewrites the ratchet and API-surface records; tests/test_architecture.py is what checks them.
guardrails:
    JAX_PLATFORMS=cpu uv run --no-sync python tools/guardrails.py --write

smoke:
    uv run --no-sync python tools/smoke_cli_workflow.py

examples:
    uv run --no-sync python examples/simulate_and_reconstruct.py

accelerator-smoke:
    uv run --no-sync python tools/smoke_accelerator.py

accelerator-smoke-cuda: (with-gpu-lock "accelerator-smoke-cuda-unlocked")

test-cuda: (with-gpu-lock "test-cuda-unlocked")

# Hold one lock for the entire suite, including its CUDA smoke check.
[private]
with-gpu-lock recipe:
    #!/bin/sh
    set -eu
    set -- {{quote(just_executable())}} --justfile {{quote(justfile())}} {{quote(recipe)}}
    if command -v flock >/dev/null 2>&1; then
        mkdir -p -- "$HOME/.cache"
        exec flock "$HOME/.cache/tomojax-gpu.lock" "$@"
    fi
    exec "$@"

[private]
accelerator-smoke-cuda-unlocked:
    TOMOJAX_REQUIRE_CUDA=1 uv run --no-sync python tools/smoke_accelerator.py

# Every test with the GPU visible: unmarked tests take the CUDA paths too, which
# CI's CPU runner and `-m gpu` alone never exercise. The workers of the second stage
# share the GPU. In about half of such runs XLA fails to create a cuFFT plan in one
# worker, which loses that worker's CUDA context and fails every later test in it, so
# the tests that fail are run again in one process.
[private]
test-cuda-unlocked: accelerator-smoke-cuda-unlocked
    uv run --no-sync pytest -q -m fresh_gpu
    uv run --no-sync pytest -q -m "not fresh_gpu" {{gpu_parallel}} || uv run --no-sync pytest -q -m "not fresh_gpu" --last-failed --last-failed-no-failures none

benchmark-smoke:
    JAX_PLATFORMS=cpu uv run --no-sync python bench/compare_projectors.py --libraries tomojax --sizes 32 --views 30 --repeats 1
    JAX_PLATFORMS=cpu uv run --no-sync python bench/reconstruction_benchmark.py --sizes 32 --views 60 --repeats 1
    JAX_PLATFORMS=cpu uv run --no-sync python bench/workflow_benchmark.py --sizes 16 --geometries parallel --methods fista spdhg --repeats 1

check: format lint typecheck imports test

[parallel]
fast-check: format-check lint fast-typecheck imports architecture-check

[private]
format-check:
    uv run --no-sync ruff format --check src tests tools examples {{benchmark_scripts}}

[private]
fast-typecheck:
    uv run --no-sync basedpyright --threads 4 --level error

[private]
architecture-check:
    JAX_PLATFORMS=cpu uv run --no-sync pytest -q tests/test_architecture.py

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
