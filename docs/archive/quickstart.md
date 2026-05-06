# Quickstart

This guide gets you from zero to an aligned reconstruction in five
steps. It uses 32^3 volumes that run in under two minutes on CPU, so
you can verify your setup before scaling up.

> [!NOTE]
> Make sure you've completed the [installation](installation.md)
> steps and verified your environment with `tomojax-test-gpu` or
> `tomojax-test-cpu` before continuing.

## 1. Simulate a phantom

Generate a small synthetic dataset with random cubes and spheres:

```bash
uv run tomojax-simulate \
  --out data/sim_aligned.nxs \
  --nx 32 --ny 32 --nz 32 \
  --nu 32 --nv 32 --n-views 32 \
  --phantom random_shapes \
  --n-cubes 8 --n-spheres 8 \
  --min-size 3 --max-size 8 \
  --min-value 0.01 --max-value 0.1 \
  --seed 1 --progress
```

This creates `data/sim_aligned.nxs`, an HDF5 file following the
[NXtomo convention](reference/data-format.md).

## 2. Misalign the projections

Apply random per-view rigid-body perturbations to simulate a
misaligned scan:

```bash
uv run tomojax-misalign \
  --data data/sim_aligned.nxs \
  --out data/sim_misaligned.nxs \
  --rot-deg 1.0 --trans-px 4 \
  --seed 0 --progress
```

## 3. Reconstruct with FBP (naive baseline)

Run filtered backprojection on the misaligned data to see the effect
of misalignment:

```bash
uv run tomojax-recon \
  --data data/sim_misaligned.nxs \
  --algo fbp --filter ramp \
  --out out/fbp_misaligned.nxs --progress
```

## 4. Align and reconstruct

Run joint multi-resolution alignment and FISTA-TV reconstruction:

```bash
uv run tomojax-align \
  --data data/sim_misaligned.nxs \
  --levels 4 2 1 \
  --outer-iters 4 --recon-iters 10 \
  --lambda-tv 0.003 \
  --opt-method gn --gn-damping 1e-3 \
  --log-summary \
  --out out/align_misaligned.nxs --progress
```

## 5. Compare results

You now have two reconstructions to compare:

- `out/fbp_misaligned.nxs` — naive FBP (blurred by misalignment)
- `out/align_misaligned.nxs` — aligned reconstruction (sharp)

Open both in your preferred HDF5 viewer (napari, tomviz, or HDFView)
and inspect the volume at
`/entry/processing/tomojax/volume`.

## Next steps

- [End-to-end tutorial](tutorials/end-to-end.md) — full 256^3
  workflow with noise, deterministic misalignment schedules, and
  parameter exports
- [CLI reference](cli/index.md) — all command options and
  configuration
- [Python API reference](reference/api.md) — use TomoJAX from
  scripts and notebooks
