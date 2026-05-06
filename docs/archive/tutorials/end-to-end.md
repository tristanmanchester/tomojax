# End-to-end tutorial

This tutorial walks you through a complete TomoJAX workflow: simulate
a 3D phantom, generate misaligned projections, add noise, run naive
FBP reconstructions, and then run iterative alignment and
reconstruction with multi-resolution coarse-to-fine refinement.

> [!NOTE]
> Make sure you've completed the [installation](../installation.md)
> and verified your environment before starting. Set
> `TOMOJAX_PROGRESS=1` for progress bars throughout.

## What you'll produce

By the end of this tutorial you'll have four `.nxs` files to compare:

- `out/fbp_misaligned.nxs` — naive FBP on misaligned data
- `out/fbp_misaligned_noisy.nxs` — naive FBP on noisy + misaligned
  data
- `out/align_misaligned.nxs` — aligned reconstruction (clean)
- `out/align_misaligned_noisy.nxs` — aligned reconstruction (noisy)

## Before you begin

Run a quick preflight on a small 32^3 volume to warm the JAX/XLA
compiler cache and verify your environment:

```bash
uv run tomojax-test-gpu
uv run tomojax-simulate \
  --out data/sim_aligned_small.nxs \
  --nx 32 --ny 32 --nz 32 --nu 32 --nv 32 --n-views 32 \
  --phantom random_shapes --n-cubes 8 --n-spheres 8 \
  --min-size 3 --max-size 8 --min-value 0.01 --max-value 0.1 \
  --seed 1 --progress
uv run tomojax-misalign \
  --data data/sim_aligned_small.nxs \
  --out data/sim_misaligned_small.nxs \
  --rot-deg 1 --trans-px 4 --seed 0 --progress
```

> [!TIP]
> The first projector/align call compiles XLA kernels and takes
> longer. Subsequent runs with the same shapes are fast.

## 1. Simulate a 256^3 phantom

Generate a phantom with 40 random cubes and 40 spheres:

```bash
uv run tomojax-simulate \
  --out data/sim_aligned.nxs \
  --nx 256 --ny 256 --nz 256 \
  --nu 256 --nv 256 --n-views 200 \
  --phantom random_shapes \
  --n-cubes 40 --n-spheres 40 \
  --min-size 4 --max-size 64 \
  --min-value 0.01 --max-value 0.1 \
  --seed 42 --progress
```

This creates a parallel-beam NXtomo dataset with geometry and grid
metadata. See the [simulate CLI](../cli/simulate.md) for all
options.

## 2. Create misaligned projections

Re-project the ground-truth volume with random per-view 5-DOF
perturbations:

```bash
uv run tomojax-misalign \
  --data data/sim_aligned.nxs \
  --out data/sim_misaligned.nxs \
  --rot-deg 1.0 --trans-px 10 \
  --seed 0 --progress
```

For deterministic drift patterns instead of random jitter, use
`--pert` schedules. See
[misalignment modes](../reference/misalign-modes.md):

```bash
uv run tomojax-misalign \
  --data data/sim_aligned.nxs \
  --out runs/mis_angle_lin.nxs \
  --pert angle:linear:delta=5deg
```

## 3. Add Poisson noise

Create a second misaligned dataset with photon noise:

```bash
uv run tomojax-misalign \
  --data data/sim_aligned.nxs \
  --out data/sim_misaligned_poisson.nxs \
  --rot-deg 1.0 --trans-px 10 \
  --poisson 100 \
  --seed 0 --progress
```

## 4. Reconstruct with FBP (naive baseline)

Run filtered backprojection on both datasets to see the effect of
misalignment:

```bash
uv run tomojax-recon \
  --data data/sim_misaligned.nxs \
  --algo fbp --filter ramp \
  --gather-dtype bf16 --checkpoint-projector \
  --out out/fbp_misaligned.nxs --progress

uv run tomojax-recon \
  --data data/sim_misaligned_poisson.nxs \
  --algo fbp --filter ramp \
  --gather-dtype bf16 --checkpoint-projector \
  --out out/fbp_misaligned_noisy.nxs --progress
```

## 5. Align and reconstruct (multi-resolution)

Run joint alignment with Gauss-Newton updates and a 3-level
coarse-to-fine pyramid:

```bash
# Clean misaligned data
uv run tomojax-align \
  --data data/sim_misaligned.nxs \
  --levels 4 2 1 \
  --outer-iters 4 --recon-iters 25 --lambda-tv 0.003 \
  --opt-method gn --gn-damping 1e-3 \
  --gather-dtype bf16 --checkpoint-projector \
  --log-summary \
  --out out/align_misaligned.nxs --progress

# Noisy + misaligned data (stronger TV)
uv run tomojax-align \
  --data data/sim_misaligned_poisson.nxs \
  --levels 4 2 1 \
  --outer-iters 5 --recon-iters 30 \
  --lambda-tv 0.03 --tv-prox-iters 20 \
  --opt-method gn --gn-damping 1e-3 \
  --gather-dtype bf16 --checkpoint-projector \
  --log-summary \
  --out out/align_misaligned_noisy.nxs --progress
```

> [!TIP]
> For larger scans, use SPDHG-TV as the inner solver to process
> stochastic view subsets:
>
> ```bash
> uv run tomojax-align \
>   --data data/sim_misaligned_poisson.nxs \
>   --levels 4 2 1 \
>   --outer-iters 5 --recon-iters 120 --lambda-tv 0.03 \
>   --recon-algo spdhg --views-per-batch 16 \
>   --spdhg-seed 0 --recon-positivity \
>   --opt-method gn --gn-damping 1e-3 \
>   --gather-dtype bf16 --checkpoint-projector \
>   --log-summary \
>   --out out/align_noisy_spdhg.nxs --progress
> ```

## 6. Compare results

You now have four reconstructions. Open them in your preferred HDF5
viewer (napari, tomviz, HDFView) and inspect the volume at
`/entry/processing/tomojax/volume`.

Each aligned `.nxs` file also contains the estimated per-view
alignment parameters at `/entry/processing/tomojax/align/thetas`.

## Key tuning parameters

| Parameter | Flag | Effect |
|-----------|------|--------|
| Gather precision | `--gather-dtype bf16` | Reduces bandwidth; accumulation stays fp32 |
| Projector checkpointing | `--checkpoint-projector` | Trades ~10-25% compute for lower memory |
| TV strength | `--lambda-tv` | Increase for noisy data |
| TV prox iterations | `--tv-prox-iters` | Increase (20-30) for heavy noise |
| Optimizer | `--opt-method gn` | GN is robust for L2 losses; try `lbfgs` for SSIM/Charbonnier |

## Data I/O shortcuts

Validate a file:

```bash
uv run tomojax-validate data/sim_aligned.nxs
```

Convert between `.npz` and `.nxs`:

```bash
uv run tomojax-convert --in data/sim_aligned.nxs \
  --out data/sim_aligned.npz
```

## Next steps

- [Laminography tutorial](laminography.md) — tilted rotation-axis
  geometry
- [Alignment concepts](../concepts/alignment.md) — algorithm
  background
- [align CLI reference](../cli/align.md) — all flags and config
  options
