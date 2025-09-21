# CLI Reference

All commands run inside the pixi environment. Either use `python -m tomojax.cli.<cmd>` or the pixi tasks (`pixi run <cmd>`) which forward extra arguments.

Common tips
- Show progress bars: add `--progress` or set `TOMOJAX_PROGRESS=1`.
- Keep `--checkpoint-projector` on for lower memory, and consider `--gather-dtype bf16`.
- Disable JAX preallocation to reduce spikes: `export XLA_PYTHON_CLIENT_PREALLOCATE=false`.


## simulate

Generate a synthetic dataset and save to `.nxs` (HDF5 NXtomo).

Usage
```
python -m tomojax.cli.simulate --out <path.nxs> \
  --nx <int> --ny <int> --nz <int> \
  --nu <int> --nv <int> --n-views <int> \
  [--geometry parallel|lamino] [--tilt-deg <float>] [--tilt-about x|z] \
  [--rotation-deg <float>] \
  [--phantom shepp|cube|sphere|blobs|random_shapes|lamino_disk] [phantom-args...] \
  [--noise none|gaussian|poisson] [--noise-level <float>] \
  [--seed <int>] [--progress]
```

Key options
- Geometry: `--geometry parallel` (default) or `lamino` with `--tilt-deg` (default 30) about axis `--tilt-about` (`x` default).
- Rotation span: `--rotation-deg` sets the total angular range. Defaults to 180 for `parallel` and 360 for `lamino`.
- Phantom:
  - Single centered object: `--phantom cube|sphere` with `--single-size` (relative fraction of min dim; side for cube, diameter for sphere) and `--single-value`. For `cube`, a random 3D rotation is applied by default; disable with `--no-single-rotate`.
  - Random shapes: `--phantom random_shapes` with controls:
    - `--n-cubes` (8), `--n-spheres` (7), `--min-size` (4), `--max-size` (32)
    - `--min-value` (0.1), `--max-value` (1.0), `--max-rot-deg` (180)
  - Laminography slab: `--phantom lamino_disk` with `--lamino-thickness-ratio` and the random_shapes knobs above.
- Noise: `--noise gaussian --noise-level <sigma>` or `--noise poisson --noise-level <scale>`.

Example
```
pixi run simulate \
  --out data/sim_aligned.nxs \
  --nx 256 --ny 256 --nz 256 --nu 256 --nv 256 --n-views 200 \
  --phantom random_shapes --n-cubes 40 --n-spheres 40 \
  --min-size 4 --max-size 64 --min-value 0.01 --max-value 0.1 --seed 42 --progress
```


## misalign

Create a misaligned (and optionally noisy) dataset from a ground‑truth `.nxs` that contains a volume.

Usage
```
python -m tomojax.cli.misalign --data <in.nxs> --out <out.nxs> \
  [--rot-deg <float>] [--trans-px <float>] [--poisson <float>] \
  [--pert dof:shape[:k=v[,k=v...]]] [--spec schedules.json] [--with-random] \
  [--seed <int>] [--progress]
```

Key options
- `--rot-deg`: max absolute per‑view rotation in degrees for α, β, φ (default 1.0).
- `--trans-px`: max absolute per‑view translation in detector pixels for (dx, dz) (default 10.0). Converted to world units via detector spacing.
- `--poisson`: incident intensity scale s for Poisson noise. Data are treated as intensities; noise is sampled as `Poisson(proj * s) / s`. Larger `s` → lower relative noise. Set 0 to disable.
- Deterministic schedules:
  - `--pert dof:shape[:k=v,...]` to add a schedule; repeatable. DOFs: `angle,alpha,beta,phi,dx,dz`. Shapes: `linear`, `sin-window`, `step`, `box`.
  - `--spec <json>` to load schedules from a file. See `docs/misalign_modes.md`.
  - `--with-random` to add random jitter on top of deterministic schedules (by default, schedules alone are used when present).

Examples
```
# Clean misalignment
pixi run misalign --data data/sim_aligned.nxs --out data/sim_misaligned.nxs \
  --rot-deg 1.0 --trans-px 10 --seed 0 --progress

# Misalignment + Poisson noise
pixi run misalign --data data/sim_aligned.nxs --out data/sim_misaligned_poisson.nxs \
  --rot-deg 1.0 --trans-px 10 --poisson 5000 --seed 0 --progress

# Deterministic schedules (see docs/misalign_modes.md)
# Linear angle drift 0→+5° across the scan
pixi run misalign --data data/sim_aligned.nxs --out runs/mis_angle_lin.nxs \
  --pert angle:linear:delta=5deg
# Sudden dx shift of +5 px at 90° (held to end)
pixi run misalign --data data/sim_aligned.nxs --out runs/mis_dx_step.nxs \
  --pert dx:step:at=90deg,to=5px
```


## recon

Reconstruct a volume from projections via FBP or FISTA‑TV. Saves a new `.nxs` with the reconstruction.

Usage
```
python -m tomojax.cli.recon --data <in.nxs> \
  [--algo fbp|fista] [--filter ramp|shepp|hann] \
  [--iters <int>] [--lambda-tv <float>] [--tv-prox-iters <int>] [--L <float>] \
  [--gather-dtype fp32|bf16|fp16] [--checkpoint-projector|--no-checkpoint-projector] \
  --out <out.nxs> [--frame sample|lab] [--progress]
```

Key options
- Algorithm: `--algo fbp` (default) or `fista`.
- FBP filter: `--filter ramp` (aliases: ram‑lak/ramlak), `shepp`, `hann`.
- FISTA: `--iters` (50), `--lambda-tv` (0.005), `--tv-prox-iters` (10) controls the inner TV proximal iterations (use 20–30 for heavy noise), optional fixed `--L` to skip power‑method.
- Memory/performance: use `--gather-dtype` (bf16 recommended on modern GPUs) and keep projector checkpointing on by default.
- Frame: `--frame sample` (default; recommended) records that the saved volume is in the sample/object frame. `lab` is recorded for compatibility exports.

Examples
```
# FBP with bf16 gathers
pixi run recon --data data/sim_misaligned.nxs \
  --algo fbp --filter ramp --gather-dtype bf16 \
  --checkpoint-projector --out out/fbp_misaligned.nxs --progress

# FISTA with TV (streamed)
pixi run recon --data data/sim_misaligned.nxs \
  --algo fista --iters 60 --lambda-tv 0.005 \
  --gather-dtype bf16 --checkpoint-projector \
  --out out/fista_misaligned.nxs --progress
```


## align

Joint per‑view alignment and reconstruction (alternating FISTA‑TV and alignment updates). Supports single‑level or multi‑resolution.

Usage
```
python -m tomojax.cli.align --data <in.nxs> \
  [--outer-iters <int>] [--recon-iters <int>] [--lambda-tv <float>] \
  [--tv-prox-iters <int>] \
  [--lr-rot <float>] [--lr-trans <float>] \
  [--opt-method gd|gn] [--gn-damping <float>] \
  [--early-stop|--no-early-stop] [--early-stop-rel <float>] [--early-stop-patience <int>] \
  [--w-rot <float>] [--w-trans <float>] [--seed-translations] \
  [--levels <ints...>] [--gather-dtype fp32|bf16|fp16] \
  [--checkpoint-projector|--no-checkpoint-projector] [--recon-L <float>] [--log-summary] \
  --out <out.nxs> [--progress]
```

Key options
- Outer/inner loops: `--outer-iters` (5), `--recon-iters` (10), `--lambda-tv` (0.005), `--tv-prox-iters` (10; increase to 20–30 for noisy data).
- Alignment step: gradient descent (`--lr-rot`, `--lr-trans`) or Gauss‑Newton (`--opt-method gn`, `--gn-damping`).
- Early stopping (alignment across outers):
  - Enable/disable: `--early-stop` (default) or `--no-early-stop`.
  - Threshold and patience: `--early-stop-rel` (default 1e-3), `--early-stop-patience` (default 2).
- Smoothness across views: `--w-rot`, `--w-trans` (default 1e‑3 in CLI).
- Multi‑resolution: `--levels 4 2 1` for coarse→fine; optional `--seed-translations` uses phase correlation at the coarsest level.
- Memory/performance: same knobs as recon (gather dtype and checkpointing).
- `--recon-L`: fixes the Lipschitz constant to skip per‑level power‑method if you already know a good bound.
- Early stopping and GN acceptance are enabled by default during alignment: outers stop when relative improvement is tiny, and GN steps are rejected if they don’t reduce loss.
  Use `--log-summary` to see when early stopping triggers.

Notes
- The align CLI initializes a persistent JAX compilation cache automatically. Set `TOMOJAX_JAX_CACHE_DIR` to control the cache location.

Examples
```
# GN, multires
pixi run align --data data/sim_misaligned.nxs \
  --levels 4 2 1 --outer-iters 4 --recon-iters 25 --lambda-tv 0.003 \
  --opt-method gn --gn-damping 1e-3 \
  --gather-dtype bf16 --checkpoint-projector \
  --log-summary --out out/align_misaligned.nxs --progress

# GD with learning rates (single level)
pixi run align --data data/sim_misaligned.nxs \
  --outer-iters 6 --recon-iters 30 --lambda-tv 0.005 \
  --opt-method gd --lr-rot 3e-3 --lr-trans 1e-1 \
  --out out/align_gd.nxs --progress
```


## convert

Convert between `.npz` and `.nxs` (HDF5). In `.nxs`, data follow the NXtomo convention with TomoJAX extras. See `docs/schema_nxtomo.md`.

Usage
```
python -m tomojax.cli.convert --in <in.npz|in.nxs> --out <out.nxs|out.npz>
```

Examples
```
pixi run python -m tomojax.cli.convert --in data/sim_aligned.nxs --out data/sim_aligned.npz
pixi run python -m tomojax.cli.convert --in data/sim_aligned.npz --out data/sim_aligned_back.nxs
```


## Environment variables

- `TOMOJAX_PROGRESS=1` — enable progress bars.
- `TOMOJAX_JAX_CACHE_DIR=<path>` — persistent JAX compilation cache directory.
- `JAX_PLATFORM_NAME=cpu` or `JAX_PLATFORMS=cuda` — pin backend selection.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` — disable device memory preallocation.
