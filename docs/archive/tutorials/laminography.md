# Laminography tutorial

This tutorial covers laminography — CT with a tilted rotation axis.
You'll simulate a thin-slab phantom, run FBP as a sanity check,
create misaligned variants, and then align and reconstruct.

> [!NOTE]
> Laminography uses a different geometry from standard parallel-beam
> CT. The rotation axis is tilted away from the beam direction,
> requiring 360-degree scans for full coverage. See
> [Geometry concepts](../concepts/geometry.md) for background.

## Before you begin

Make sure your environment is set up:

```bash
# GPU
uv sync --extra cuda12 --group dev

# CPU only
uv sync --extra cpu --group dev
```

Set `TOMOJAX_PROGRESS=1` for progress bars.

## 1. Simulate a laminography dataset

Create a 128^3 thin-slab phantom with tilted geometry:

```bash
uv run tomojax-simulate \
  --out runs/lamino_demo.nxs \
  --nx 128 --ny 128 --nz 128 \
  --nu 128 --nv 128 --n-views 360 \
  --geometry lamino \
  --tilt-deg 35 --tilt-about x \
  --phantom lamino_disk \
  --lamino-thickness-ratio 0.15 \
  --n-cubes 100 --n-spheres 100 \
  --min-size 4 --max-size 32 \
  --seed 3
```

The ground-truth volume is saved in the sample frame (object
coordinates). The default rotation span for laminography is 360
degrees.

## 2. Reconstruct the clean dataset (sanity check)

Run FBP to verify the simulation:

```bash
uv run tomojax-recon \
  --data runs/lamino_demo.nxs \
  --algo fbp --filter ramp \
  --out runs/lamino_demo_fbp.nxs
```

## 3. Create misaligned datasets

Apply random per-view perturbations:

```bash
# Clean misalignment
uv run tomojax-misalign \
  --data runs/lamino_demo.nxs \
  --out runs/lamino_demo_misaligned.nxs \
  --rot-deg 5.0 --trans-px 10 \
  --seed 0 --progress

# Misalignment + Poisson noise
uv run tomojax-misalign \
  --data runs/lamino_demo.nxs \
  --out runs/lamino_demo_misaligned_noisy.nxs \
  --rot-deg 5.0 --trans-px 10 \
  --poisson 10 \
  --seed 0 --progress
```

For deterministic drift patterns, use `--pert` schedules. See
[misalignment modes](../reference/misalign-modes.md):

```bash
# Linear angle drift across 360-degree scan
uv run tomojax-misalign \
  --data runs/lamino_demo.nxs \
  --out runs/lamino_mis_angle_lin.nxs \
  --pert angle:linear:delta=5deg

# dz box pulse between 60 and 80 degrees
uv run tomojax-misalign \
  --data runs/lamino_demo.nxs \
  --out runs/lamino_mis_dz_box.nxs \
  --pert dz:box:at=60deg,width_deg=20,delta=-4px
```

## 4. Reconstruct misaligned data (naive FBP)

```bash
uv run tomojax-recon \
  --data runs/lamino_demo_misaligned.nxs \
  --algo fbp --filter ramp \
  --out runs/lamino_demo_misaligned_fbp.nxs

uv run tomojax-recon \
  --data runs/lamino_demo_misaligned_noisy.nxs \
  --algo fbp --filter ramp \
  --out runs/lamino_demo_misaligned_noisy_fbp.nxs
```

## 5. Align and reconstruct

```bash
# Clean misaligned
uv run tomojax-align \
  --data runs/lamino_demo_misaligned.nxs \
  --outer-iters 4 --recon-iters 10 \
  --lambda-tv 5e-3 --tv-prox-iters 10 \
  --opt-method gn --gn-damping 1e-3 \
  --out runs/lamino_demo_aligned.nxs \
  --progress --log-summary

# Noisy + misaligned (stronger TV)
uv run tomojax-align \
  --data runs/lamino_demo_misaligned_noisy.nxs \
  --outer-iters 4 --recon-iters 20 \
  --lambda-tv 5e-2 --tv-prox-iters 15 \
  --opt-method gn --gn-damping 1e-3 \
  --out runs/lamino_demo_noisy_aligned.nxs \
  --progress --log-summary
```

> [!TIP]
> Reconstructions are saved in the sample frame. Slices advance
> parallel to the rotation axis (the thin slab is visible in each
> slice). If the orientation looks wrong in your viewer, check the
> `@frame` attribute and consider transposing axes.

## Next steps

- [Geometry concepts](../concepts/geometry.md) — laminography
  geometry details
- [Alignment concepts](../concepts/alignment.md) — tuning the
  optimizer
- [align CLI reference](../cli/align.md) — all flags and examples
