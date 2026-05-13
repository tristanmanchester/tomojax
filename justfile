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

production-surface-check:
    uv run ruff format --check src/tomojax/align/api.py src/tomojax/align/__init__.py src/tomojax/bench/api.py src/tomojax/bench/__init__.py src/tomojax/calibration/api.py src/tomojax/calibration/__init__.py src/tomojax/cli/main.py src/tomojax/cli/ingest.py src/tomojax/cli/recon.py src/tomojax/cli/align.py src/tomojax/cli/misalign.py src/tomojax/cli/loss_bench.py src/tomojax/io tests/test_cli_public_surface.py tests/test_io_public_dataset.py tests/test_public_facades.py tests/test_cli_entrypoints.py
    uv run ruff check src/tomojax/cli
    uv run ruff check --select I,TID,D100,E402,RUF022 src/tomojax/align/api.py src/tomojax/align/__init__.py src/tomojax/bench/api.py src/tomojax/bench/__init__.py src/tomojax/calibration/api.py src/tomojax/calibration/__init__.py src/tomojax/cli/main.py src/tomojax/cli/ingest.py src/tomojax/cli/recon.py src/tomojax/cli/align.py src/tomojax/cli/misalign.py src/tomojax/cli/loss_bench.py src/tomojax/io tests/test_cli_public_surface.py tests/test_io_public_dataset.py tests/test_public_facades.py tests/test_cli_entrypoints.py
    uv run basedpyright src/tomojax/io src/tomojax/cli/main.py src/tomojax/cli/ingest.py src/tomojax/cli/_runtime.py src/tomojax/cli/config.py
    uv run lint-imports --config .importlinter
    uv run python tools/check_public_imports.py
    uv run pytest tests/test_cli_public_surface.py tests/test_io_public_dataset.py tests/test_public_facades.py tests/test_convert.py tests/test_validate_cli.py tests/test_inspect_cli.py tests/test_simulate.py::test_simulate_cli_builds_config_and_calls_simulate_to_file tests/test_simulate.py::test_simulate_cli_incomplete_explicit_artefacts_preserve_legacy_noise tests/test_simulate.py::test_simulate_cli_rejects_invalid_explicit_artefact tests/test_preprocess.py::test_preprocess_cli_smoke tests/test_preprocess.py::test_preprocess_cli_combines_crop_reject_and_auto_reject tests/test_align_auto_cli.py::test_public_cli_scripts_use_single_grouped_entrypoint tests/test_cli_entrypoints.py tests/test_bench_alignment_smoke.py::test_alignment_smoke_in_process_align_preserves_cli_shape tests/test_small_module_coverage.py::test_convert_main_parses_paths_and_calls_convert tests/test_misalign_schedules.py tests/test_loss_bench.py -q

bench-smoke:
    uv run python -m tomojax.datasets generate --manifest docs/tomojax-v2/benchmark_manifest.yaml --size 32 --out .artifacts/smoke
    uv run python -m tomojax.verify synthetic-smoke .artifacts/smoke

bench-128:
    uv run python -m tomojax.datasets generate --manifest docs/tomojax-v2/benchmark_manifest.yaml --size 128 --out .artifacts/bench128
    uv run python -m tomojax.benchmarks.run .artifacts/bench128
