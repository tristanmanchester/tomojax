# Repository Guidelines (TomoJAX v2)

## Project Structure & Module Organization
- Core projector and operators live under `src/tomojax/core/`.
  - Pose‑aware forward projector: `src/tomojax/core/projector.py`
  - Geometry types: `src/tomojax/core/geometry/`
- Reconstruction lives under `src/tomojax/recon/` (FBP, FISTA‑TV, multires helpers).
- Alignment workflows live under `src/tomojax/align/` (see `pipeline.py`).
- Reusable benchmark datasets and scoring helpers live under `src/tomojax/bench/`.
- CLI entry points live under `src/tomojax/cli/`:
  - `simulate`, `misalign`, `preprocess`, `recon`, `align`, `inspect`, `validate`, `convert`, `loss_bench`.
- Tests live in `tests/` (CPU‑friendly sizes).
- Input data and generated artifacts belong in `data/` or `runs/`; these are git‑ignored.
- Figures for docs live in `images/`.

## Build, Test, and Development Commands
- Sync the managed environment with `uv sync --extra cuda12 --group dev` (or `uv sync --extra cpu --group dev` for CPU-only).
- Benchmark-harness runs under `bench/` need `uv sync --extra bench --group dev` in addition to the normal CPU/GPU extra.
- Verify JAX/accelerator: `uv run tomojax-test-gpu` (prints backend and devices). For CPU: `JAX_PLATFORM_NAME=cpu uv run tomojax-test-cpu`.
- Run tests: `uv run pytest -q tests` or target a file, e.g., `uv run pytest -q tests/test_projector.py`.
- CLI workflows (examples; see `README.md` for more):
  - Simulate: `uv run tomojax-simulate --help`
  - Recon: `uv run tomojax-recon --data data/sim.nxs --algo fbp --out runs/fbp.nxs`
  - Align: `uv run tomojax-align --data data/sim_misaligned.nxs --levels 2 1 --outer-iters 4 --out runs/align.nxs`

## Coding Style & Naming Conventions
- Python 3.12, 4‑space indents, aim for ≤100 characters per line, prefer `jnp.float32` arrays.
- Provide type hints and concise docstrings on public helpers; keep JAX code functional and side‑effect free.
- Follow snake_case for functions/variables, UPPER_SNAKE_CASE for constants; extend APIs with kwargs instead of breaking signatures (e.g., projector functions).

## Testing Guidelines
- Test surface ownership and placement rules live in `tests/README.md`; use that guide when deciding whether a change needs a unit, workflow, CLI, or benchmark-harness regression.
- Mirror existing patterns in `tests/test_*.py`; keep problem sizes CPU‑friendly.
- Name tests `test_*` and guard expensive cases behind flags or explicit CLI switches.
- Record CLI commands and seeds when running experiments; capture them in PR descriptions or notes.

## Commit & Pull Request Guidelines
- Use Conventional Commits (`feat:`, `fix:`, `chore:`, etc.) with imperative subjects ≤72 characters.
- PRs should state the problem, link issues, list reproduction commands, detail expected vs. actual results, and include performance notes (device, shapes, runtime).
- Attach before/after visuals from `images/` or run outputs when changes affect reconstructions or docs.

## Security & Configuration Tips
- CUDA 12 on Linux is required for GPU runs; install the `cuda12` extra when syncing with uv.
- Never commit large datasets or generated artifacts; verify `.gitignore` coverage before pushing.
- JAX persistent compilation cache: the `align` CLI enables it if available; override location via `TOMOJAX_JAX_CACHE_DIR` (defaults to `~/.cache/tomojax/jax_cache`).
