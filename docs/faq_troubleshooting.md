# FAQ & Troubleshooting

## Environment & Installation

- GPU not detected or wrong backend:
  - Check: `pixi run test-gpu` (inside `pixi shell`).
  - Force CPU: `JAX_PLATFORM_NAME=cpu pixi run test-cpu`.
  - Ensure NVIDIA driver and CUDA 12 runtime compatible with your hardware.

- Slow first run: JAX/XLA compilation is cached; subsequent runs are faster.
  - Enable a persistent cache for repeated runs: set `TOMOJAX_JAX_CACHE_DIR` (or rely on the default at `~/.cache/tomojax/jax_cache`).


## Memory / OOM

- Out of memory (RESOURCE_EXHAUSTED) during FBP or FISTA:
  - Use `--views-per-batch auto` (recommended) or a small integer (e.g., 4 or 8).
  - Reduce `--projector-unroll` to 1–2.
  - Keep `--checkpoint-projector` enabled.
  - Prefer `--gather-dtype bf16` on newer GPUs.
  - Disable JAX preallocation: `export XLA_PYTHON_CLIENT_PREALLOCATE=false`.

- Auto batch seems too large:
  - Clamp with `export TOMOJAX_MAX_VIEWS_PER_BATCH=4`.
  - For very large volumes (≥256³), expect conservative caps by design.


## Alignment Convergence

- Poor progress under gradient descent:
  - Switch to Gauss–Newton: `--opt-method gn --gn-damping 1e-3`.
  - Increase outer iterations: `--outer-iters`.
  - Strengthen TV for noisy data: increase `--lambda-tv`.

- Coarse-to-fine strategy:
  - Use `--levels` (e.g., `4 2 1`).
  - Optionally seed translations at the coarsest level: `--seed-translations`.


## Data I/O

- Validate an `.nxs` file quickly:
  ```bash
  python - << 'PY'
  from tomojax.data.io_hdf5 import validate_nxtomo
  print(validate_nxtomo('path/to/data.nxs'))
  PY
  ```

- Convert between `.npz` and `.nxs`:
  ```bash
  pixi run python -m tomojax.cli.convert --in data/sim.npz --out data/sim.nxs
  pixi run python -m tomojax.cli.convert --in data/sim.nxs --out data/sim.npz
  ```


## Reproducibility

- Record the exact CLI and seeds used for alignment experiments.
- Example:
  ```bash
  pixi run python -m tomojax.cli.align \
    --data data/sim_misaligned.nxs \
    --levels 4 2 1 --outer-iters 15 --lambda-tv 0.005 \
    --opt-method gn --gn-damping 1e-3 --seed-translations
  ```

