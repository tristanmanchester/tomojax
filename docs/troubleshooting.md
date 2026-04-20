# Troubleshooting

This page covers common issues you may encounter when using TomoJAX,
organized by category. If your issue isn't listed here, check the
[CLI overview](cli/index.md) for environment variables and flag
defaults.

## Installation and environment

### GPU not detected or wrong backend

Run the verification utility:

```bash
uv run tomojax-test-gpu
```

If it reports `cpu` instead of `gpu`:

1. Verify your NVIDIA driver: `nvidia-smi`
2. Verify CUDA 12 is installed and compatible with your hardware
3. Make sure you installed with the GPU extra:
   `uv sync --extra cuda12 --group dev`

To force a specific backend:

```bash
# Force CPU
JAX_PLATFORM_NAME=cpu uv run tomojax-test-cpu

# Force CUDA
JAX_PLATFORMS=cuda uv run tomojax-test-gpu
```

### CUDA_ERROR_INVALID_IMAGE or "Failed to load in-memory CUBIN"

These errors mean cached XLA kernels were compiled for a different
GPU. Clear all caches and rerun:

```bash
rm -rf ~/.cache/tomojax ~/.cache/jax ~/.cache/xla ~/.nv/ComputeCache
```

XLA rebuilds kernels for your current device on the next run. If you
need a persistent cache, set `TOMOJAX_JAX_CACHE_DIR` after the
matching kernels are regenerated.

### Slow first run

JAX/XLA compiles kernels on first use. This is expected.

Enable a persistent compilation cache to speed up subsequent runs:

```bash
export TOMOJAX_JAX_CACHE_DIR=~/.cache/tomojax/jax_cache
```

The `align` CLI enables this cache automatically at the default
location.

## Memory and OOM

### RESOURCE_EXHAUSTED during FBP or FISTA

Try these steps in order:

1. Keep `--checkpoint-projector` enabled (on by default)
2. Use `--gather-dtype bf16` on GPUs that support bfloat16
3. Disable JAX memory preallocation:
   `export XLA_PYTHON_CLIENT_PREALLOCATE=false`
4. Reduce problem size (coarser `--levels`, fewer views, smaller
   grid)

### Estimating memory before a run

Use `tomojax-inspect` to see memory estimates without running any
computation:

```bash
uv run tomojax-inspect data/scan.nxs
```

The output includes estimated memory for FBP, FISTA-TV, and SPDHG-TV
at fp32 precision. Use these estimates to decide whether your GPU has
enough memory.

## Alignment convergence

### Poor progress under gradient descent

Switch to Gauss-Newton, which converges faster on L2-like losses:

```bash
--opt-method gn --gn-damping 1e-3
```

If GN isn't available for your chosen loss, try L-BFGS:

```bash
--opt-method lbfgs --lbfgs-maxiter 20
```

### Alignment stalls or oscillates

Try these adjustments:

- Increase `--outer-iters` to give the optimizer more steps
- Increase `--lambda-tv` for noisy data to improve the intermediate
  reconstruction quality
- Increase `--tv-prox-iters` to 20-30 for heavy noise
- Use multi-resolution (`--levels 4 2 1`) to resolve large
  misalignments at coarse levels first
- Seed translations at the coarsest level:
  `--seed-translations`

### Interpreting log-summary output

Add `--log-summary` to see per-outer-iteration summaries:

- **FISTA**: first/last/min objective values
- **GN alignment**: loss before/after, mean step magnitudes, and
  whether the step was accepted or rejected
- **L-BFGS alignment**: objective values, iteration counts, and
  fallback status
- **Early stopping**: logged when triggered

### Choosing a loss schedule

For translation-only (2-DOF) coarse-to-fine alignment:

```bash
--optimise-dofs dx,dz \
--loss-schedule 4:phasecorr,2:ssim,1:l2_otsu
```

For full 5-DOF alignment, start conservatively:

```bash
--loss ssim
```

See [Loss functions reference](reference/loss-functions.md) for the
full list and recommendations.

## Data I/O

### Validating a file

Run the validation CLI:

```bash
uv run tomojax-validate data/scan.nxs
```

Or use the Python function:

```python
from tomojax.data.io_hdf5 import validate_nxtomo
report = validate_nxtomo("data/scan.nxs")
print(report)
```

### Converting between formats

```bash
uv run tomojax-convert --in data/scan.nxs --out data/scan.npz
uv run tomojax-convert --in data/scan.npz --out data/scan_back.nxs
```

### Volume orientation looks wrong in a viewer

TomoJAX saves volumes in `(nz, ny, nx)` on-disk order with
`@volume_axes_order="zyx"`. Some viewers assume a different axis
order.

- Check the `@frame` attribute at
  `/entry/processing/tomojax@frame` — it records whether the volume
  is in the sample or lab frame
- For laminography, slices advance parallel to the rotation axis
- The `load_nxtomo()` function always transposes to internal
  `(nx, ny, nz)` order

## Laminography

### Why 360 degrees?

Parallel-beam laminography benefits from full 360-degree coverage.
The default rotation span for `--geometry lamino` is 360 degrees
(vs 180 for standard parallel-beam).

### No alignment progress shown

Use `--progress --log-summary` together. Keep progress bars visible
by setting `TOMOJAX_PROGRESS_LEAVE=1`.

## Reproducibility

### Recording experiment details

Pass `--save-manifest manifest.json` to write a JSON sidecar with:

- Raw CLI arguments
- Resolved config (after TOML + flag merge)
- TomoJAX, Python, and JAX versions
- JAX backend and device list
- UTC timestamp

### Resuming interrupted alignment

Use checkpointing to save and resume long runs:

```bash
# Start with checkpointing
uv run tomojax-align --data data/scan.nxs \
  --checkpoint out/align.checkpoint.npz \
  --checkpoint-every 1 \
  --out out/align.nxs

# Resume after interruption
uv run tomojax-align --data data/scan.nxs \
  --resume out/align.checkpoint.npz \
  --out out/align.nxs
```

Checkpoints are saved at outer-iteration boundaries, not mid-FISTA.

## Next steps

- [CLI overview](cli/index.md) — environment variables and common
  flags
- [Alignment concepts](concepts/alignment.md) — algorithm details
- [Installation](installation.md) — full setup guide
