# Laminography End-to-End Tutorial

This guide focuses on laminography: simulate a thin slab, run FBP for a sanity check, generate misaligned variants (clean and noisy), then align and reconstruct.

Prerequisites
- `uv sync --extra cuda12 --group dev` completed
- Optional: set JAX_PLATFORM_NAME=cpu if CUDA libraries aren’t available

## Simulate laminography dataset
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

Notes
- Saves the ground-truth volume in the sample frame (object coordinates).
- For laminography, the default rotation span is 360°.

## Recon clean dataset with FBP - sanity check
```bash
uv run tomojax-recon \
--data runs/lamino_demo.nxs \
--algo fbp \
--filter ramp \
--out runs/lamino_demo_fbp.nxs
```

## Create misaligned dataset
```bash
uv run tomojax-misalign \
  --data runs/lamino_demo.nxs \
  --out runs/lamino_demo_misaligned.nxs \
  --rot-deg 5.0 --trans-px 10 \
  --seed 0 \
  --progress
```

Deterministic misalignment schedules
- For systematic drifts/steps, use `--pert`/`--spec` (see `docs/misalign_modes.md`). Examples:
```bash
# Angle linear drift 0→+5° across 360° scan
uv run tomojax-misalign --data runs/lamino_demo.nxs --out runs/lamino_mis_angle_lin.nxs \
  --pert angle:linear:delta=5deg
# dz box pulse −4 px between ~60° and ~80°
uv run tomojax-misalign --data runs/lamino_demo.nxs --out runs/lamino_mis_dz_box.nxs \
  --pert dz:box:at=60deg,width_deg=20,delta=-4px
```

## Create misaligend noisy dataset
```bash
uv run tomojax-misalign \
  --data runs/lamino_demo.nxs \
  --out runs/lamino_demo_misaligned_noisy.nxs \
  --rot-deg 5.0 --trans-px 10 \
  --seed 0 \
  --poisson 10 \
  --progress
```

## Naive FBP reconstruction misaligned dataset
```bash
uv run tomojax-recon \
  --data runs/lamino_demo_misaligned.nxs \
  --algo fbp \
  --filter ramp \
  --out runs/lamino_demo_misaligned_fbp.nxs
```

## Naive FBP reconstruction misaligned noisy dataset
```bash
uv run tomojax-recon \
  --data runs/lamino_demo_misaligned_noisy.nxs \
  --algo fbp \
  --filter ramp \
  --out runs/lamino_demo_misaligned_noisy_fbp.nxs
```

## Align and reconstruct misaligned dataset
```bash
uv run tomojax-align \
  --data runs/lamino_demo_misaligned.nxs \
  --outer-iters 4 --recon-iters 10 \
  --lambda-tv 5e-3 --tv-prox-iters 10 \
  --opt-method gn --gn-damping 1e-3 \
  --out runs/lamino_demo_aligned.nxs \
  --progress --log-summary
```

## Align and reconstruct misaligned noisy dataset
```bash
uv run tomojax-align \
  --data runs/lamino_demo_misaligned_noisy.nxs \
  --outer-iters 4 --recon-iters 20 \
  --lambda-tv 5e-2 --tv-prox-iters 15 \
  --opt-method gn --gn-damping 1e-3 \
  --out runs/lamino_demo_noisy_aligned.nxs \
  --progress --log-summary
```

Tips
- Reconstructions are saved in the sample frame; use consistent axis order when viewing.
- For memory-constrained runs, keep projector checkpointing enabled.
