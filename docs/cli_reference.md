# CLI Reference

All commands run inside the pixi environment. Either use `python -m tomojax.cli.<cmd>` or the pixi tasks (`pixi run <cmd>`) which forward extra arguments.

Common tips
- Show progress bars: add `--progress` or set `TOMOJAX_PROGRESS=1`.
- Memory safety: prefer `--views-per-batch auto`, keep `--checkpoint-projector` on, and consider `--gather-dtype bf16`.
- Disable JAX preallocation to reduce spikes: `export XLA_PYTHON_CLIENT_PREALLOCATE=false`.


## simulate

Generate a synthetic dataset and save to `.nxs` (HDF5 NXtomo).

Usage
```
python -m tomojax.cli.simulate --out <path.nxs> \
  --nx <int> --ny <int> --nz <int> \
  --nu <int> --nv <int> --n-views <int> \
  [--geometry parallel|lamino] [--tilt-deg <float>] [--tilt-about x|z] \
  [--phantom shepp|cube|blobs|random_shapes] [phantom-args...] \
  [--noise none|gaussian|poisson] [--noise-level <float>] \
  [--seed <int>] [--progress]
```

Key options
- Geometry: `--geometry parallel` (default) or `lamino` with `--tilt-deg` (default 30) about axis `--tilt-about` (`x` default).
- Phantom: `--phantom random_shapes` with controls:
  - `--n-cubes` (8), `--n-spheres` (7), `--min-size` (4), `--max-size` (32)
  - `--min-value` (0.1), `--max-value` (1.0), `--max-rot-deg` (180)
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
  [--seed <int>] [--progress]
```

Key options
- `--rot-deg`: max absolute per‑view rotation in degrees for α, β, φ (default 1.0).
- `--trans-px`: max absolute per‑view translation in detector pixels for (dx, dz) (default 10.0). Converted to world units via detector spacing.
- `--poisson`: incident intensity scale s for Poisson noise. Data are treated as intensities; noise is sampled as `Poisson(proj * s) / s`. Larger `s` → lower relative noise. Set 0 to disable.

Examples
```
# Clean misalignment
pixi run misalign --data data/sim_aligned.nxs --out data/sim_misaligned.nxs \
  --rot-deg 1.0 --trans-px 10 --seed 0 --progress

# Misalignment + Poisson noise
pixi run misalign --data data/sim_aligned.nxs --out data/sim_misaligned_poisson.nxs \
  --rot-deg 1.0 --trans-px 10 --poisson 5000 --seed 0 --progress
```


## recon

Reconstruct a volume from projections via FBP or FISTA‑TV. Saves a new `.nxs` with the reconstruction.

Usage
```
python -m tomojax.cli.recon --data <in.nxs> \
  [--algo fbp|fista] [--filter ramp|shepp|hann] \
  [--iters <int>] [--lambda-tv <float>] [--L <float>] \
  [--views-per-batch <int|auto>] [--projector-unroll <int>] \
  [--gather-dtype fp32|bf16|fp16] [--checkpoint-projector|--no-checkpoint-projector] \
  --out <out.nxs> [--progress]
```

Key options
- Algorithm: `--algo fbp` (default) or `fista`.
- FBP filter: `--filter ramp` (aliases: ram‑lak/ramlak), `shepp`, `hann`.
- FISTA: `--iters` (50), `--lambda-tv` (0.005), optional fixed `--L` to skip power‑method.
- Memory/performance: `--views-per-batch auto` (recommended), `--projector-unroll` (1–4), `--gather-dtype` (bf16 recommended on modern GPUs), projector checkpointing on by default.

Examples
```
# FBP with auto batching and bf16 gathers
pixi run recon --data data/sim_misaligned.nxs \
  --algo fbp --filter ramp --views-per-batch auto --gather-dtype bf16 \
  --checkpoint-projector --out out/fbp_misaligned.nxs --progress

# FISTA with TV
pixi run recon --data data/sim_misaligned.nxs \
  --algo fista --iters 60 --lambda-tv 0.005 \
  --views-per-batch auto --gather-dtype bf16 --checkpoint-projector \
  --out out/fista_misaligned.nxs --progress
```


## align

Joint per‑view alignment and reconstruction (alternating FISTA‑TV and alignment updates). Supports single‑level or multi‑resolution.

Usage
```
python -m tomojax.cli.align --data <in.nxs> \
  [--outer-iters <int>] [--recon-iters <int>] [--lambda-tv <float>] \
  [--lr-rot <float>] [--lr-trans <float>] \
  [--opt-method gd|gn] [--gn-damping <float>] \
  [--w-rot <float>] [--w-trans <float>] [--seed-translations] \
  [--levels <ints...>] [--views-per-batch <int|auto>] \
  [--projector-unroll <int>] [--gather-dtype fp32|bf16|fp16] \
  [--checkpoint-projector|--no-checkpoint-projector] [--recon-L <float>] [--log-summary] \
  --out <out.nxs> [--progress]
```

Key options
- Outer/inner loops: `--outer-iters` (5), `--recon-iters` (10), `--lambda-tv` (0.005).
- Alignment step: gradient descent (`--lr-rot`, `--lr-trans`) or Gauss‑Newton (`--opt-method gn`, `--gn-damping`).
- Smoothness across views: `--w-rot`, `--w-trans` (default 1e‑3 in CLI).
- Multi‑resolution: `--levels 4 2 1` for coarse→fine; optional `--seed-translations` uses phase correlation at the coarsest level.
- Memory/performance: same knobs as recon; prefer `--views-per-batch auto`.
- `--recon-L`: fixes the Lipschitz constant to skip per‑level power‑method if you already know a good bound.

Examples
```
# GN, multires, auto batching
pixi run align --data data/sim_misaligned.nxs \
  --levels 4 2 1 --outer-iters 4 --recon-iters 25 --lambda-tv 0.003 \
  --opt-method gn --gn-damping 1e-3 \
  --views-per-batch auto --gather-dtype bf16 --checkpoint-projector --projector-unroll 4 \
  --log-summary --out out/align_misaligned.nxs --progress

# GD with learning rates (single level)
pixi run align --data data/sim_misaligned.nxs \
  --outer-iters 6 --recon-iters 30 --lambda-tv 0.005 \
  --opt-method gd --lr-rot 3e-3 --lr-trans 1e-1 \
  --views-per-batch auto --out out/align_gd.nxs --progress
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
- `TOMOJAX_MAX_VIEWS_PER_BATCH=<int>` — upper clamp for `--views-per-batch auto` (default 8; auto further tightens for very large volumes).
- `TOMOJAX_JAX_CACHE_DIR=<path>` — persistent JAX compilation cache directory.
- `JAX_PLATFORM_NAME=cpu` or `JAX_PLATFORMS=cuda` — pin backend selection.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` — disable device memory preallocation.

