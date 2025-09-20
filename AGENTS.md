# Repository Guidelines (TomoJAX v2)

## Project Structure & Module Organization
- Core projector and operators live under `src/tomojax/core/`.
  - Pose‑aware forward projector: `src/tomojax/core/projector.py`
  - Geometry types: `src/tomojax/core/geometry/`
- Reconstruction lives under `src/tomojax/recon/` (FBP, FISTA‑TV, multires helpers).
- Alignment workflows live under `src/tomojax/align/` (see `pipeline.py`).
- CLI entry points live under `src/tomojax/cli/`:
  - `simulate`, `misalign`, `recon`, `align`.
- Tests live in `tests/` (CPU‑friendly sizes).
- Input data and generated artifacts belong in `data/` or `runs/`; these are git‑ignored.
- Figures for docs live in `images/`.

## Build, Test, and Development Commands
- Enter the managed environment with `pixi shell`; run all commands below from there.
- Verify JAX/accelerator: `pixi run test-gpu` (prints backend and devices). For CPU: `pixi run test-cpu`.
- Install the package in editable mode: `pixi run install-root`.
- Run tests: `pixi run test` or target a file, e.g., `pixi run python -m pytest -q tests/test_projector.py`.
- CLI workflows (examples; see `README.md` for more):
  - Simulate: `pixi run simulate --help`
  - Recon: `pixi run recon --data data/sim.nxs --algo fbp --out runs/fbp.nxs`
  - Align: `pixi run align --data data/sim_misaligned.nxs --levels 2 1 --outer-iters 4 --out runs/align.nxs`

## Coding Style & Naming Conventions
- Python 3.12, 4‑space indents, aim for ≤100 characters per line, prefer `jnp.float32` arrays.
- Provide type hints and concise docstrings on public helpers; keep JAX code functional and side‑effect free.
- Follow snake_case for functions/variables, UPPER_SNAKE_CASE for constants; extend APIs with kwargs instead of breaking signatures (e.g., projector functions).

## Testing Guidelines
- Mirror existing patterns in `tests/test_*.py`; keep problem sizes CPU‑friendly.
- Name tests `test_*` and guard expensive cases behind flags or explicit CLI switches.
- Record CLI commands and seeds when running experiments; capture them in PR descriptions or notes.

## Commit & Pull Request Guidelines
- Use Conventional Commits (`feat:`, `fix:`, `chore:`, etc.) with imperative subjects ≤72 characters.
- PRs should state the problem, link issues, list reproduction commands, detail expected vs. actual results, and include performance notes (device, shapes, runtime).
- Attach before/after visuals from `images/` or run outputs when changes affect reconstructions or docs.

## Security & Configuration Tips
- CUDA 12 is required; JAX is installed via `pixi.toml` using the `cuda12` extra.
- Never commit large datasets or generated artifacts; verify `.gitignore` coverage before pushing.
- JAX persistent compilation cache: the `align` CLI enables it if available; override location via `TOMOJAX_JAX_CACHE_DIR` (defaults to `~/.cache/tomojax/jax_cache`).
