set dotenv-load := false

# Keep temporary builds and test files off /tmp unless the caller chose a location.
export TMPDIR := if env("TMPDIR", "") == "" {
    `mkdir -p -- "$HOME/.cache/tomojax-tmp" && printf '%s' "$HOME/.cache/tomojax-tmp"`
} else {
    env("TMPDIR")
}

benchmark_scripts := "bench"
studio_toolchain := "1.99.1"

# CPU tests run on one worker per two logical CPUs, each pinned to its share of the cores
# (tests/conftest.py) and handed one test at a time, the slow ones first.
parallel := "-n " + `echo $(( ($(nproc) + 1) / 2 ))` + " --maxschedchunk 1"

# Each GPU worker needs a CPU and about 1.3 GiB on the first card.
# tests/conftest.py pins each worker and caps its JAX pool.
gpu_parallel := "-n " + `(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo 0) | awk -v cpus="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)" 'NR == 1 { by_mem = int($1 / 1300); print (by_mem < cpus ? by_mem : cpus) }'` + " --maxschedchunk 1"

setup *flags:
    ./tools/worktree_setup.sh {{flags}}

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

studio-wheel:
    rm -rf dist-studio
    cargo build --locked --release --manifest-path studio/Cargo.toml --package tomojax-studio --package tomojax-worker --features static-hdf5 --target-dir studio/target
    uv run --no-sync python tools/build_studio_wheel.py --out-dir dist-studio --check-static
    uv run --no-sync twine check --strict dist-studio/*.whl

[env("RUSTUP_TOOLCHAIN", studio_toolchain)]
studio-check: studio-lint
    cargo build --locked --manifest-path studio/Cargo.toml -p tomojax-worker
    cargo test --locked --manifest-path studio/Cargo.toml --workspace

# Build Studio and capture every workspace headless into .artifacts/studio-snap.
[env("RUSTUP_TOOLCHAIN", studio_toolchain)]
studio-verify:
    uv run --no-sync python tools/verify_studio.py --build

[private]
[env("RUSTUP_TOOLCHAIN", studio_toolchain)]
studio-lint:
    cargo fmt --all --manifest-path studio/Cargo.toml --check
    cargo clippy --locked --manifest-path studio/Cargo.toml --workspace --all-targets -- -D warnings

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

# Run every test with CUDA visible, including unmarked accelerator paths.
# The fresh-GPU tests run first. Retry failures from the shared-worker pass in one process.
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

# Independent readers for freshly exported volumes; Java artifacts are cached locally.
interop *args:
    JAX_PLATFORMS=cpu uv run --locked --group interop python tools/check_export_readers.py {{args}}
