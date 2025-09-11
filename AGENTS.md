# Repository Guidelines

## Project Structure & Module Organization
- `projector_parallel_jax.py`: Core differentiable parallel-beam CT projector (JAX).
- `alignment-testing/`: Multi‑resolution alignment scripts (`run_alignment.py`, `run_quick_test.py`, utils).
- `examples/`: Small runnable demos (projection, reconstruction, data generation).
- `data/` and run output dirs (e.g., `misaligned_test/`): Input/outputs not committed by default.
- `images/`: Figures and GIFs used in docs.

## Build, Test, and Development Commands
- Environment: `pixi` (Python 3.12, CUDA 12). Create shell: `pixi shell`.
- GPU check: `pixi run test-gpu` (prints backend and devices).
- Quick algorithm sanity test: `pixi run python alignment-testing/run_quick_test.py`.
- Full alignment run: `pixi run python alignment-testing/run_alignment.py --outer-iters 15 --lambda-tv 0.005`.
- Basic projector demo: `pixi run python examples/run_parallel_projector.py`.

## Coding Style & Naming Conventions
- Language: Python; 4‑space indents; max line length ≈ 100.
- Use type hints and docstrings; follow snake_case for functions/variables and UPPER_SNAKE_CASE for constants.
- Keep JAX code pure and functional: avoid side effects, keep arrays `jnp.float32` unless justified.
- Preserve public API signatures (e.g., `forward_project_view`, `batch_forward_project`). Add kwargs rather than breaking params; document `static_argnames` stability for `@jit`.

## Testing Guidelines
- Prefer fast checks first: `alignment-testing/run_quick_test.py`.
- Validate GPU availability with `pixi run test-gpu` before performance tests.
- For alignment experiments, record the exact CLI used and seed. Example: `pixi run python alignment-testing/run_alignment.py --bin-factors 4 2 1 --outer-iters 15`.
- If adding tests, mirror existing patterns in `alignment-testing/` (e.g., `test_*.py`) and keep sizes small for CPU.

## Commit & Pull Request Guidelines
- Use Conventional Commits where possible: `feat:`, `fix:`, `perf:`, `docs:`, `refactor:`, `test:`, `build:`.
- Subject in imperative mood, ≤ 72 chars; body explains rationale and user‑visible effects.
- PRs must include: concise description, linked issues, reproduction commands, expected vs actual, and performance notes (CPU/GPU, device, shapes).
- For visual changes, attach before/after images from `images/` or run outputs.

## Security & Configuration Tips
- CUDA 12 required for GPU; JAX installed via `pixi.toml` with `cuda12` extra.
- Optional: warm JAX cache for speedups: `bash alignment-testing/setup_jax_cache.sh`.
- Do not commit large datasets or generated outputs; keep them under `data/` or run‑specific folders listed in `.gitignore`.
