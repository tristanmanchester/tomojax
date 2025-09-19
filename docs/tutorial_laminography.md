# Laminography End-to-End Tutorial

This guide focuses on laminography: simulate a thin slab, run FBP for a sanity check, generate misaligned variants (clean and noisy), then align and reconstruct.

Prerequisites
- pixi installed and environment set up
- Optional: set JAX_PLATFORM_NAME=cpu if CUDA libraries aren’t available

## Simulate laminography dataset
```bash
pixi run simulate \
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
pixi run python -m tomojax.cli.recon \
--data runs/lamino_demo.nxs \
--algo fbp \
--filter ramp \
--views-per-batch auto \
--out runs/lamino_demo_fbp.nxs
```

## Create misaligned dataset
```bash
pixi run misalign \
  --data runs/lamino_demo.nxs \
  --out runs/lamino_demo_misaligned.nxs \
  --rot-deg 5.0 --trans-px 10 \
  --seed 0 \
  --progress
```

## Create misaligend noisy dataset
```bash
pixi run misalign \
  --data runs/lamino_demo.nxs \
  --out runs/lamino_demo_misaligned_noisy.nxs \
  --rot-deg 5.0 --trans-px 10 \
  --seed 0 \
  --poisson 10 \
  --progress
```

## Naive FBP reconstruction misaligned dataset
```bash
pixi run python -m tomojax.cli.recon \
  --data runs/lamino_demo_misaligned.nxs \
  --algo fbp \
  --filter ramp \
  --views-per-batch auto \
  --out runs/lamino_demo_misaligned_fbp.nxs
```

## Naive FBP reconstruction misaligned noisy dataset
```bash
pixi run python -m tomojax.cli.recon \
  --data runs/lamino_demo_misaligned_noisy.nxs \
  --algo fbp \
  --filter ramp \
  --views-per-batch auto \
  --out runs/lamino_demo_misaligned_noisy_fbp.nxs
```

## Align and reconstruct misaligned dataset
```bash
pixi run align \
  --data runs/lamino_demo_misaligned.nxs \
  --outer-iters 4 --recon-iters 10 \
  --lambda-tv 5e-3 --tv-prox-iters 10 \
  --opt-method gn --gn-damping 1e-3 \
  --views-per-batch auto \
  --out runs/lamino_demo_aligned.nxs \
  --progress --log-summary
```

## Align and reconstruct misaligned noisy dataset
```bash
pixi run align \
  --data runs/lamino_demo_misaligned_noisy.nxs \
  --outer-iters 4 --recon-iters 20 \
  --lambda-tv 5e-2 --tv-prox-iters 15 \
  --opt-method gn --gn-damping 1e-3 \
  --views-per-batch auto \
  --out runs/lamino_demo_noisy_aligned.nxs \
  --progress --log-summary
```

Tips
- Reconstructions are saved in the sample frame; use consistent axis order when viewing.
- For memory-constrained runs, use --views-per-batch auto and keep projector checkpointing enabled.
