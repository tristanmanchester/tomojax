# TomoJAX benchmark harness

This directory contains the fixed benchmark harness used by the remote Runpod controller.

Reusable benchmark helpers that are shared with packaged CLIs live under
`src/tomojax/bench/`. The `bench/` tree is the controller-facing harness: it owns
profiles, fixture orchestration, and reporting around those reusable helpers.

During optimisation work, prefer changing TomoJAX source code and shared benchmark support
under `src/tomojax/bench/` before editing the controller harness. Reach for changes in
`bench/` only when the benchmark contract itself needs to move.

## Why this exists

TomoJAX already has useful CLIs and experiments, but an optimisation loop needs a smaller,
more repeatable contract:

- a named profile;
- a deterministic synthetic dataset or prebuilt fixture;
- separate first-run and warm-run timings;
- JAX synchronisation via `.block_until_ready()`;
- one stable metrics JSON.

This harness provides that contract.

## Setup

Benchmark-harness runs need the benchmark extra in addition to the normal dev environment:

```bash
uv sync --extra bench --extra cuda12 --group dev
```

For CPU-only benchmark work, replace `--extra cuda12` with `--extra cpu`.

## Profile tiers

### Smoke

- `smoke`: tiny reconstruction sanity check. Use this only for remote-environment validation.
- `smoke_align`: tiny alignment sanity check using the tracked `align_small_v1.npz` fixture.

### Screening suite

These are the profiles the controller should use in the inner optimisation loop. They are large
 enough to exercise realistic 3D workloads, but still lean enough for repeated runs on cheaper
 24 GB to 48 GB GPUs.

- `screen_speed_parallel_fbp_128`: `128^3`, 192-view parallel-beam FBP speed screen.
- `screen_memory_parallel_fista_128`: `128^3`, 160-view parallel-beam FISTA-TV memory screen.
- `screen_accuracy_align_parallel_3d_96`: `96^3`, 144-view noisy 3D alignment screen.
- `screen_convergence_align_parallel_3d_96`: `96^3`, 144-view noisy 3D alignment time-to-threshold screen.
- `screen_ttq_memguard_align_parallel_3d_96`: `96^3`, 144-view noisy 3D alignment screen scored by time to a finest-level quality contract with a soft memory guard.
- `screen_ttq_memguard_align_parallel_3d_96_holdout`: same workload as the TTQ screen, different seeds for hidden holdout confirmation.
- `screen_memory_sentinel_align_parallel_3d_192_fine`: larger-shape fine-level memory sentinel used to catch bad scaling.

### Canary suite

These are slower confirmation profiles. Do not use them as the main search loop; use them to
 validate promising candidates from the screening suite.

- `canary_iterative_parallel_160`: `160^3`, 220-view iterative 3D reconstruction.
- `canary_lamino_fbp_128`: `128^3`, 320-view laminography reconstruction.
- `canary_align_parallel_3d_128_noisy`: `128×128×96`, 180-view noisy 3D alignment.
- `canary_convergence_align_parallel_3d_128_noisy`: `128×128×96`, 180-view noisy 3D alignment time-to-threshold confirmation.
- `canary_ref_align_parallel_3d_128_noisy`: operational TTQ canary with one scored warm run and coherent summary output.
- `canary_measure_align_parallel_3d_128_noisy`: repeated measurement canary for close calls and significance checks.

All objectives are lower-is-better.

## Fixtures vs generated data

The repo still ships a tiny tracked smoke fixture under `bench/fixtures/`, but the larger,
more representative profiles intentionally generate persistent benchmark datasets under
`bench/data/` on first use. Set `TOMOJAX_BENCH_DATA_ROOT` to move these generated fixtures
outside the checkout, which is the preferred setup for remote workers using ephemeral clones.

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

Representative alignment profiles write a compact PNG summary next to the metrics JSON. The image
shows central `xy`/`xz`/`yz` slices for the ground-truth volume, a nominal-geometry FBP baseline
from the misaligned projections, the final aligned volume, the absolute error volume, and a small
loss-history or convergence panel. Convergence profiles draw the tracked quality metric against
outer iteration and mark the configured threshold crossing when it happens. The metrics JSON records this via `summary_image_path` and
`summary_image_error`.

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
- `gpu_memory_process_source` and `gpu_memory_process_supported` make it explicit whether process-scoped memory came from NVML, `nvidia-smi --query-compute-apps`, or had to fall back to device-level memory.
- Set `measurement.save_jax_device_memory_profile: true` to write a JAX device-memory profile next to the metrics JSON for debugging.
- `smoke_align` disables visualization by default because it is only an infra sanity check over the tiny tracked fixture.
- Representative alignment profiles enable visualization by default, and they already use the richer `random_shapes` phantom family from generated benchmark data.
- Alignment profiles support `visualization.enabled: false` if you need to suppress the summary PNG for a specific run.
- Alignment benchmarks now support synthetic observation noise, so the accuracy screens are closer to real reconstruction/alignment use.
- Convergence-mode alignment profiles add a `convergence:` block. `threshold_scope: finest_only` means the reported crossing only counts once the finest level satisfies the threshold, even if coarser levels cross it earlier.
- TTQ profiles add `quality_contract:` and `objective_policy:` blocks. These produce `quality_contract_met`, `warm_seconds_to_quality_contract`, `memory_guard_*`, and `objective_time_memguard`.
- Versioned JSON files under `bench/reference/` freeze the calibration thresholds and memory caps used by the TTQ screen, sentinel, and operational canary.
- `align.outer_iters` and `align.recon_iters` remain the maximum work budget for convergence profiles. The run stops at the first outer iteration whose measured quality crosses the threshold when `convergence.stop_on_threshold: true`.
- Use the operational canary (`canary_ref_*`) for normal promotion decisions and keep the repeated measurement canary (`canary_measure_*`) for significance checks. Their summaries should not be compared as if they were the same benchmark mode.
