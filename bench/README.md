# TomoJAX benchmark harness

This directory contains the fixed benchmark harness used by the remote Runpod controller.

The rule is simple: during optimisation work, treat everything under `bench/` as read-only.
The optimisation target is TomoJAX source code, not the benchmark itself.

## Why this exists

TomoJAX already has useful CLIs and experiments, but an optimisation loop needs a smaller,
more repeatable contract:

- a named profile;
- a fixed fixture or tiny synthetic dataset;
- separate first-run and warm-run timings;
- JAX synchronisation via `.block_until_ready()`;
- one stable metrics JSON.

This harness provides that contract.

## Profiles

- `smoke`: tiny reconstruction smoke test. Cheap environment sanity check.
- `speed_recon_small`: representative small reconstruction benchmark. Objective: `warm_run_seconds_mean`.
- `memory_recon_small`: same reconstruction task, but objective: `peak_gpu_memory_mb`.
- `accuracy_align_small`: small alignment benchmark. Objective: `gt_mse`.

All objectives are lower-is-better.

## Fixtures vs generated data

The repo ships small tracked fixtures under `bench/fixtures/` so benchmark runs do not have to
regenerate tomography data and accidentally warm the same projector path before the first timed run.

If a fixture is missing, `bench/fitness.py` can regenerate it from the YAML profile and write it
back to the fixture path or to `bench/data/`, but that fallback should be treated as a bootstrap
path, not the normal benchmark path.

## Outputs

Metrics are written wherever `--out` points, for example:

```bash
uv run python bench/fitness.py --profile smoke --out bench/out/smoke.json
uv run python bench/fitness.py --profile speed_recon_small --out bench/out/metrics.json
uv run python bench/fitness.py --profile memory_recon_small --out bench/out/metrics.json
uv run python bench/fitness.py --profile accuracy_align_small --out bench/out/metrics.json
```

The JSON schema is documented in `bench/metrics_schema.json`.

## Remote Pod usage

The controller repo is expected to run commands like:

```bash
cd /workspace/tomojax
export JAX_COMPILATION_CACHE_DIR=/workspace/.jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run python bench/fitness.py --profile speed_recon_small --out /workspace/results/speed_recon_small.json
```

On Runpod, keep persistent data under `/workspace`, especially:

- `/workspace/.jax_cache`
- `/workspace/results`
- `/workspace/tomojax`

## Notes on the measurements

- JAX timings are invalid unless the computation is synchronised. This harness blocks before stopping the timer.
- First-run and warm-run timings are separated.
- GPU memory uses lightweight `nvidia-smi` polling when available, so it should be treated as an approximation.
