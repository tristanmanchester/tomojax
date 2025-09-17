# Repository Guidelines

## Project Structure & Module Organization
- Core differentiable projector lives in `projector_parallel_jax.py`; keep new kernels near related helpers.
- Multi-resolution alignment workflows reside in `alignment-testing/` (`run_alignment.py`, `run_quick_test.py`, utilities`).
- Runnable demos stay in `examples/`; treat them as reference patterns for new scripts.
- Input data and generated artifacts belong in `data/` or run-specific folders (e.g., `misaligned_test/`); these paths are git-ignored.
- Figures for docs live in `images/`; mirror existing naming before adding assets.

## Build, Test, and Development Commands
- Enter the managed environment with `pixi shell`; run all commands below from there.
- Confirm accelerator availability via `pixi run test-gpu` (reports backend and devices).
- Smoke-test the alignment pipeline: `pixi run python alignment-testing/run_quick_test.py`.
- Launch a full alignment pass: `pixi run python alignment-testing/run_alignment.py --outer-iters 15 --lambda-tv 0.005`.
- Validate projector basics using `pixi run python examples/run_parallel_projector.py`.

## Coding Style & Naming Conventions
- Python 3.12, 4-space indents, aim for ≤100 characters per line, prefer `jnp.float32` arrays.
- Provide type hints and concise docstrings on public helpers; keep JAX code functional and side-effect free.
- Follow snake_case for functions/variables, UPPER_SNAKE_CASE for constants; extend APIs with kwargs instead of breaking signatures (`forward_project_view`, `batch_forward_project`).

## Testing Guidelines
- Mirror existing patterns in `alignment-testing/test_*.py`; keep problem sizes CPU-friendly.
- Name tests `test_*` and guard expensive cases behind flags or explicit CLI switches.
- Record CLI commands and seeds when running experiments; capture them in PR descriptions or notes.

## Commit & Pull Request Guidelines
- Use Conventional Commits (`feat:`, `fix:`, etc.) with imperative subjects ≤72 characters.
- PRs should state the problem, link issues, list reproduction commands, detail expected vs. actual results, and include performance notes (device, shapes, runtime).
- Attach before/after visuals from `images/` or run outputs when changes affect reconstructions or docs.

## Security & Configuration Tips
- CUDA 12 is required; JAX installs via `pixi.toml` using the `cuda12` extra.
- Never commit large datasets or generated artifacts; double-check `.gitignore` coverage before pushing.
- Optionally warm the JAX compilation cache with `bash alignment-testing/setup_jax_cache.sh` after dependency updates.
