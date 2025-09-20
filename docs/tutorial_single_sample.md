# TomoJAX Next — Single‑Sample (Centered Cube/Sphere) Tutorial

This guide mirrors the end‑to‑end tutorial but uses a single centered object — a
cube or sphere — as the phantom. You will:

1) simulate a centered single‑object phantom, 2) create a misaligned forward model,
3) optionally add noise, 4) run naive FBP, 5) run iterative alignment + reconstruction.

All commands assume you are inside the pixi environment:

- Enter the environment: `pixi shell`
- One‑time install of the package: `pixi run install-root`
- Show progress bars: `export TOMOJAX_PROGRESS=1`


## 0) Quick Warmup (optional)

```
pixi run test-gpu
pixi run simulate \
  --out data/sim_single_small.nxs \
  --nx 32 --ny 32 --nz 32 \
  --nu 32 --nv 32 --n-views 32 \
  --phantom sphere --single-size 0.6 --single-value 1.0 \
  --seed 1 \
  --progress
pixi run misalign --data data/sim_single_small.nxs --out data/sim_single_misaligned_small.nxs --rot-deg 1 --trans-px 4 --seed 0 --progress
```

Notes
- First projector/align call compiles with XLA; subsequent runs are faster.


## 1) Simulate a 256³ Single‑Object Phantom

Choose `sphere` (default below) or `cube`. The `--single-size` parameter is a
relative fraction of the minimum dimension: for a sphere it is the diameter; for
a cube it is the side length. The object is centered in the volume. For `cube`, a
random 3D rotation is applied by default for a more interesting sample (disable
with `--no-single-rotate`).

```
pixi run simulate \
  --out data/sim_single.nxs \
  --nx 256 --ny 256 --nz 256 \
  --nu 256 --nv 256 --n-views 200 \
  --phantom sphere \
  --single-size 0.3 --single-value 1.0 \
  --seed 42 \
  --progress
```

Alternative (cube): (random 3D rotation by default)

```
pixi run simulate \
  --out data/sim_single.nxs \
  --nx 256 --ny 256 --nz 256 \
  --nu 256 --nv 256 --n-views 200 \
  --phantom cube \
  --single-size 0.3 --single-value 0.3 \
  --seed 69 \
  --progress
# To force an axis-aligned cube instead, add --no-single-rotate
```

Notes
- Geometry defaults to parallel‑beam; detector matches the volume size.
- Datasets are saved as NeXus (NXtomo) with geometry/grid metadata.


## 2) Create a Misaligned Forward Model

Apply per‑view perturbations to create a misaligned dataset:

```
pixi run misalign \
  --data data/sim_single.nxs \
  --out data/sim_single_misaligned.nxs \
  --rot-deg 3.0 --trans-px 5 \
  --seed 0 \
  --progress

Deterministic misalignment schedules
- For systematic drifts/steps, use `--pert` and/or `--spec` (see `docs/misalign_modes.md`). Examples:

```
# Angle linear drift 0→+5° across the scan
pixi run misalign --data data/sim_single.nxs --out runs/single_mis_angle_lin.nxs \
  --pert angle:linear:delta=5deg

# dx sinusoidal drift peaking +5 px at mid‑scan
pixi run misalign --data data/sim_single.nxs --out runs/single_mis_dx_sin.nxs \
  --pert dx:sin-window:amp=5px
```
```


## 3) Add Poisson Noise (optional)

```
pixi run misalign \
  --data data/sim_single.nxs \
  --out data/sim_single_misaligned_poisson.nxs \
  --rot-deg 3.0 --trans-px 5 \
  --poisson 10 \
  --seed 0 \
  --progress
```


## 4) Naive FBP Reconstructions

```
# Misaligned
pixi run recon \
  --data data/sim_single_misaligned.nxs \
  --algo fbp --filter ramp \
  --gather-dtype bf16 --checkpoint-projector \
  --out out/fbp_single_misaligned.nxs \
  --progress

# Noisy + misaligned
pixi run recon \
  --data data/sim_single_misaligned_poisson.nxs \
  --algo fbp --filter ramp \
  --gather-dtype bf16 --checkpoint-projector \
  --out out/fbp_single_misaligned_noisy.nxs \
  --progress
```


## 5) Iterative Alignment + Reconstruction (Levels [4, 2, 1])

```
# Misaligned (clean)
pixi run align \
  --data data/sim_single_misaligned.nxs \
  --levels 4 2 1 \
  --outer-iters 10 --recon-iters 15 --lambda-tv 0.003 \
  --opt-method gn --gn-damping 1e-3 \
  --gather-dtype bf16 --checkpoint-projector \
  --log-summary \
  --out out/align_single_misaligned.nxs \
  --progress

# Noisy + misaligned
pixi run align \
  --data data/sim_single_misaligned_poisson.nxs \
  --levels 4 2 1 \
  --outer-iters 5 --recon-iters 10 --lambda-tv 0.1 --tv-prox-iters 10 \
  --opt-method gn --gn-damping 1e-3 \
  --gather-dtype bf16 --checkpoint-projector \
  --log-summary \
  --out out/align_single_misaligned_noisy.nxs \
  --progress
```

Tips
- Keep projector checkpointing enabled and prefer `--gather-dtype bf16` when on GPU.
- Increase `--lambda-tv` and `--tv-prox-iters` for heavy noise.


## 6) Outputs

Compare:
- Naive FBP: `out/fbp_single_misaligned.nxs`, `out/fbp_single_misaligned_noisy.nxs`
- Aligned: `out/align_single_misaligned.nxs`, `out/align_single_misaligned_noisy.nxs`

Aligned outputs include per‑view alignment parameters.


## Appendix — Key Flags

- `--phantom {sphere,cube}` with `--single-size` (diameter/side as fraction of min dim) and `--single-value`.
- `--gather-dtype {auto,fp32,bf16,fp16}`; `bf16` recommended on modern GPUs.
- `--[no-]checkpoint-projector` to trade compute for memory.
- `--opt-method {gn,gd}`; GN is typically faster to high quality.
