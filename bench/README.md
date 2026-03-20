# TomoJAX benchmark harness

This directory contains the fixed benchmark harness used by the remote Runpod controller.

During optimisation work, treat everything under `bench/` as read-only. The optimisation
 target is TomoJAX source code, not the benchmark itself.

## Why this exists

TomoJAX already has useful CLIs and experiments, but an optimisation loop needs a smaller,
more repeatable contract:

- a named profile;
- a deterministic synthetic dataset or prebuilt fixture;
- separate first-run and warm-run timings;
- JAX synchronisation via `.block_until_ready()`;
- one stable metrics JSON.

This harness provides that contract.

## Profile tiers

### Smoke

- `smoke`: tiny reconstruction sanity check. Use this only for remote-environment validation.

### Screening suite

These are the profiles the controller should use in the inner optimisation loop. They are large
 enough to exercise realistic 3D workloads, but still lean enough for repeated runs on cheaper
 24 GB to 48 GB GPUs.

- `screen_speed_parallel_fbp_128`: `128^3`, 192-view parallel-beam FBP speed screen.
- `screen_memory_parallel_fista_128`: `128^3`, 160-view parallel-beam FISTA-TV memory screen.
- `screen_accuracy_align_parallel_3d_96`: `96^3`, 144-view noisy 3D alignment screen.

### Canary suite

These are slower confirmation profiles. Do not use them as the main search loop; use them to
 validate promising candidates from the screening suite.

- `canary_iterative_parallel_160`: `160^3`, 220-view iterative 3D reconstruction.
- `canary_lamino_fbp_128`: `128^3`, 320-view laminography reconstruction.
- `canary_align_parallel_3d_128_noisy`: `128×128×96`, 180-view noisy 3D alignment.

All objectives are lower-is-better.

## Fixtures vs generated data

The repo still ships a tiny tracked smoke fixture under `bench/fixtures/`, but the larger,
more representative profiles intentionally generate persistent benchmark datasets under
`bench/data/` on first use.

That keeps the repository light while still giving each remote worker a stable dataset after its
first preparation run. The benchmark timing starts only after fixture generation, so the metrics
JSON is not polluted by dataset creation time.

Normal remote runs should therefore:

- keep `bench/data/` persistent on the Pod volume;
- treat the generated `.npz` bundles as read-only once created;
- exclude `bench/data/` from controller syncs so workers keep their cached fixtures.

## Outputs

Metrics are written wherever `--out` points, for example:

```bash
uv run python bench/fitness.py --profile smoke --out bench/out/smoke.json
uv run python bench/fitness.py --profile screen_speed_parallel_fbp_128 --out bench/out/speed.json
uv run python bench/fitness.py --profile screen_memory_parallel_fista_128 --out bench/out/memory.json
uv run python bench/fitness.py --profile screen_accuracy_align_parallel_3d_96 --out bench/out/align.json
```

The JSON schema is documented in `bench/metrics_schema.json`.

## Remote Pod usage

The controller repo is expected to run commands like:

```bash
cd /workspace/tomojax
export JAX_COMPILATION_CACHE_DIR=/workspace/.jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run python bench/fitness.py \
  --profile screen_speed_parallel_fbp_128 \
  --out /workspace/results/screen_speed_parallel_fbp_128.json
```

On Runpod, keep persistent data under `/workspace`, especially:

- `/workspace/.jax_cache`
- `/workspace/results`
- `/workspace/tomojax`
- `/workspace/tomojax/bench/data`

## Measurement notes

- JAX timings are invalid unless the computation is synchronised. This harness blocks before stopping the timer.
- First-run and warm-run timings are separated.
- GPU memory uses NVML sampling inside the benchmark process.
- `peak_gpu_memory_mb` uses process-scoped GPU memory when NVML process queries are available, with device-level fallback recorded separately.
- Set `measurement.save_jax_device_memory_profile: true` to write a JAX device-memory profile next to the metrics JSON for debugging.
- Alignment benchmarks now support synthetic observation noise, so the accuracy screens are closer to real reconstruction/alignment use.
