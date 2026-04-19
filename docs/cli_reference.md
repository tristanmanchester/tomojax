# CLI Reference

All commands run inside the uv-managed environment. Either use `python -m tomojax.cli.<cmd>` or the console scripts via `uv run tomojax-<cmd>`.

Common tips
- Show progress bars: add `--progress` or set `TOMOJAX_PROGRESS=1`.
- Keep `--checkpoint-projector` on for lower memory, and consider `--gather-dtype bf16`.
- Disable JAX preallocation to reduce spikes: `export XLA_PYTHON_CLIENT_PREALLOCATE=false`.

Config files
- `tomojax-recon` and `tomojax-align` support `--config <path.toml>` for reproducible long commands.
- Precedence is `built-in defaults < TOML config file < explicit CLI flags`.
- Config keys are flat top-level TOML keys matching argparse destination names: use underscores, not dashes, for example `lambda_tv`, `views_per_batch`, `checkpoint_projector`, `save_manifest`.
- Lists use TOML arrays, for example `grid = [128, 128, 128]`, `levels = [4, 2, 1]`, and `loss_param = ["delta=1.0", "eps=0.001"]`.
- Booleans use TOML booleans, for example `checkpoint_projector = true` or `progress = false`.
- YAML is not supported by the runtime CLI because `pyyaml` is only an optional benchmark dependency. Use TOML (`.toml`) instead.
- Example files: `docs/recon_config.toml` and `docs/align_config.toml`.


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
uv run tomojax-simulate \
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
uv run tomojax-misalign --data data/sim_aligned.nxs --out data/sim_misaligned.nxs \
  --rot-deg 1.0 --trans-px 10 --seed 0 --progress

# Misalignment + Poisson noise
uv run tomojax-misalign --data data/sim_aligned.nxs --out data/sim_misaligned_poisson.nxs \
  --rot-deg 1.0 --trans-px 10 --poisson 5000 --seed 0 --progress

# Deterministic schedules (see docs/misalign_modes.md)
# Linear angle drift 0→+5° across the scan
uv run tomojax-misalign --data data/sim_aligned.nxs --out runs/mis_angle_lin.nxs \
  --pert angle:linear:delta=5deg
# Sudden dx shift of +5 px at 90° (held to end)
uv run tomojax-misalign --data data/sim_aligned.nxs --out runs/mis_dx_step.nxs \
  --pert dx:step:at=90deg,to=5px
```


## recon

Reconstruct a volume from projections via FBP, FISTA‑TV, or SPDHG‑TV. Saves a new `.nxs` with the reconstruction.

Usage
```
python -m tomojax.cli.recon [--config <config.toml>] --data <in.nxs> \
  [--algo fbp|fista|spdhg] [--filter ramp|shepp|hann] \
  [--iters <int>] [--lambda-tv <float>] [--tv-prox-iters <int>] [--L <float>] \
  [--views-per-batch <int>] [--theta <float>] \
  [--spdhg-seed <int>] [--spdhg-tau <float>] [--spdhg-sigma-data <float>] [--spdhg-sigma-tv <float>] \
  [--roi off|auto|cube|bbox] [--mask-vol off|cyl] [--grid NX NY NZ] \
  [--gather-dtype fp32|bf16|fp16] [--checkpoint-projector|--no-checkpoint-projector] \
  --out <out.nxs> [--quicklook <out.png>|--save-preview <out.png>] \
  [--save-manifest <manifest.json>] [--frame sample|lab] [--progress]
```

Key options
- Config: `--config docs/recon_config.toml` loads flat TOML defaults. Explicit CLI flags override file values, so `--config cfg.toml --lambda-tv 0.01` replaces any `lambda_tv` value in the file.
- Algorithm: `--algo fbp` (default), `fista`, or `spdhg`.
- FBP filter: `--filter ramp` (aliases: ram‑lak/ramlak), `shepp`, `hann`.
- FISTA: `--iters` (50), `--lambda-tv` (0.005), `--tv-prox-iters` (10) controls the inner TV proximal iterations (use 20–30 for heavy noise), optional fixed `--L` to skip power‑method.
- SPDHG‑TV: `--iters` (outer PDHG steps), `--lambda-tv` (TV weight), `--views-per-batch` (stochastic block size, e.g. 16–64), `--theta` (extrapolation, e.g. 0.5–1.0). Step sizes default to operator‑norm‑based auto; override with `--spdhg-tau`, `--spdhg-sigma-data`, `--spdhg-sigma-tv`. Use `--spdhg-seed` to fix block order.
- Memory/performance: use `--gather-dtype` (bf16 recommended on modern GPUs) and keep projector checkpointing on by default.
- ROI/masking: `--roi auto|cube|bbox|off` to crop the recon grid to the detector FOV; `--mask-vol cyl` applies a cylindrical x–y mask (used as a support in SPDHG and post‑hoc for FBP).
- Preview: `--quicklook out.png` or `--save-preview out.png` writes a percentile-scaled central `xy` slice PNG after the `.nxs` reconstruction is saved.
- Reproducibility: `--save-manifest manifest.json` writes a JSON sidecar with raw argv, parsed CLI args, resolved config, TomoJAX/Python/JAX versions, JAX backend/devices, and a UTC timestamp.
- Frame: `--frame sample` (default; recommended) records that the saved volume is in the sample/object frame. `lab` is recorded for compatibility exports.

Examples
```
# FBP with bf16 gathers
uv run tomojax-recon --data data/sim_misaligned.nxs \
  --algo fbp --filter ramp --gather-dtype bf16 \
  --checkpoint-projector --out out/fbp_misaligned.nxs \
  --save-manifest out/fbp_misaligned.manifest.json --progress

# FBP with a central-slice quicklook PNG
uv run tomojax-recon --data data/sim_misaligned.nxs \
  --algo fbp --filter ramp \
  --out out/fbp_misaligned.nxs \
  --quicklook out/fbp_misaligned.png

# FISTA with TV (streamed)
uv run tomojax-recon --data data/sim_misaligned.nxs \
  --algo fista --iters 60 --lambda-tv 0.005 \
  --gather-dtype bf16 --checkpoint-projector \
  --out out/fista_misaligned.nxs --progress

# SPDHG‑TV with moderate block size
uv run tomojax-recon --data data/sim_aligned.nxs \
  --algo spdhg --iters 300 --lambda-tv 0.005 \
  --views-per-batch 32 --theta 0.5 \
  --gather-dtype bf16 --checkpoint-projector \
  --roi auto --mask-vol cyl \
  --out out/spdhg_aligned.nxs --progress

# Same command style from TOML, with one explicit override
uv run tomojax-recon --config docs/recon_config.toml --gather-dtype bf16
```


## align

Joint per‑view alignment and reconstruction (alternating FISTA‑TV and alignment updates). Supports single‑level or multi‑resolution.

Usage
```
python -m tomojax.cli.align [--config <config.toml>] --data <in.nxs> \
  [--outer-iters <int>] [--recon-iters <int>] [--lambda-tv <float>] \
  [--tv-prox-iters <int>] \
  [--lr-rot <float>] [--lr-trans <float>] \
  [--opt-method gd|gn] [--gn-damping <float>] \
  [--optimise-dofs <names>] [--freeze-dofs <names>] [--bounds <spec>] \
  [--early-stop|--no-early-stop] [--early-stop-rel <float>] [--early-stop-patience <int>] \
  [--pose-model per_view|polynomial|spline] [--knot-spacing <int>] [--degree <int>] \
  [--w-rot <float>] [--w-trans <float>] [--seed-translations] \
  [--levels <ints...>] [--gather-dtype fp32|bf16|fp16] \
  [--checkpoint-projector|--no-checkpoint-projector] [--recon-L <float>] [--log-summary] \
  --out <out.nxs> [--save-params-json <out.json>] [--save-params-csv <out.csv>] \
  [--save-manifest <manifest.json>] [--progress]
```

Key options
- Config: `--config docs/align_config.toml` loads flat TOML defaults. Explicit CLI flags override file values, including list values such as `--levels 2 1` and booleans such as `--no-checkpoint-projector`.
- Outer/inner loops: `--outer-iters` (5), `--recon-iters` (10), `--lambda-tv` (0.005), `--tv-prox-iters` (10; increase to 20–30 for noisy data).
- Alignment step: gradient descent (`--lr-rot`, `--lr-trans`) or Gauss‑Newton (`--opt-method gn`, `--gn-damping`).
- Active DOFs: choose from `alpha`, `beta`, `phi`, `dx`, `dz`. By default all five are optimised. Use `--optimise-dofs dx,dz` for translation-only alignment, or `--freeze-dofs phi` to keep selected parameters fixed at their initial values.
- Parameter bounds: `--bounds dx=-20:20,dz=-20:20,alpha=-0.05:0.05` clips named DOFs after each update. Rotations (`alpha`, `beta`, `phi`) are in radians; translations (`dx`, `dz`) are in world units. Omitted DOFs are unconstrained, and frozen DOFs stay fixed even if a bound is supplied for them.
- Pose model: `--pose-model per_view` (default) optimises one independent parameter vector per view. `--pose-model spline --knot-spacing N --degree 3` optimises smooth knot trajectories and expands them back to per-view parameters before projection. `--pose-model polynomial --degree D` fits each active DOF as a low-degree polynomial over the scan coordinate.
- Early stopping (alignment across outers):
  - Enable/disable: `--early-stop` (default) or `--no-early-stop`.
  - Threshold and patience: `--early-stop-rel` (default 1e-3), `--early-stop-patience` (default 2).
- Smoothness across views: `--w-rot`, `--w-trans` (default 1e‑3 in CLI) add a second-difference prior on the expanded per-view parameters. Smooth pose models usually need less explicit prior strength because the basis already reduces freedom.
- Multi‑resolution: `--levels 4 2 1` for coarse→fine; optional `--seed-translations` uses phase correlation at the coarsest level.
- Memory/performance: same knobs as recon (gather dtype and checkpointing).
- `--recon-L`: fixes the Lipschitz constant to skip per‑level power‑method if you already know a good bound.
- Parameter exports: `--save-params-json` and `--save-params-csv` write named per-view sidecars with `alpha_rad`, `beta_rad`, `phi_rad`, `dx_world`, `dz_world`, `dx_px`, and `dz_px`.
- Reproducibility: `--save-manifest manifest.json` writes a JSON sidecar with raw argv, parsed CLI args, resolved config, TomoJAX/Python/JAX versions, JAX backend/devices, and a UTC timestamp.
- Early stopping and GN acceptance are enabled by default during alignment: outers stop when relative improvement is tiny, and GN steps are rejected if they don’t reduce loss.
  Use `--log-summary` to see when early stopping triggers.

Notes
- The `.nxs` output still stores the final alignment parameters; JSON/CSV sidecars are optional convenience exports for plotting and reproducibility.
- DOF selection does not change the saved parameter format: outputs still use five columns in `[alpha, beta, phi, dx, dz]` order, with inactive columns held fixed.
- Smooth motion models are best for slow drift, stage sag, thermal/mechanical trends, or noisy data where independent per-view parameters overfit. Keep `per_view` for abrupt shifts, dropped-view artifacts, or genuinely view-local motion.
- The align CLI initializes a persistent JAX compilation cache automatically. Set `TOMOJAX_JAX_CACHE_DIR` to control the cache location.

Examples
```
# GN, multires
uv run tomojax-align --data data/sim_misaligned.nxs \
  --levels 4 2 1 --outer-iters 4 --recon-iters 25 --lambda-tv 0.003 \
  --opt-method gn --gn-damping 1e-3 \
  --gather-dtype bf16 --checkpoint-projector \
  --log-summary --out out/align_misaligned.nxs \
  --save-params-json out/align_misaligned.params.json \
  --save-params-csv out/align_misaligned.params.csv \
  --save-manifest out/align_misaligned.manifest.json \
  --progress

# GD with learning rates (single level)
uv run tomojax-align --data data/sim_misaligned.nxs \
  --outer-iters 6 --recon-iters 30 --lambda-tv 0.005 \
  --opt-method gd --lr-rot 3e-3 --lr-trans 1e-1 \
  --out out/align_gd.nxs --progress

# 2-DOF translation-only alignment
uv run tomojax-align --data data/sim_misaligned.nxs \
  --levels 4 2 1 --opt-method gn --optimise-dofs dx,dz \
  --out out/align_translation_only.nxs

# 4-DOF alignment with in-plane spin fixed
uv run tomojax-align --data data/sim_misaligned.nxs \
  --levels 4 2 1 --opt-method gn --freeze-dofs phi \
  --out out/align_no_phi.nxs

# Bounded translations and one rotation, preserving unconstrained behavior for other DOFs
uv run tomojax-align --data data/sim_misaligned.nxs \
  --levels 4 2 1 --opt-method gn \
  --bounds dx=-20:20,dz=-20:20,alpha=-0.05:0.05 \
  --out out/align_bounded.nxs

# Smooth spline model for noisy drift-like motion
uv run tomojax-align --data data/sim_misaligned_poisson.nxs \
  --levels 4 2 1 --opt-method gn \
  --pose-model spline --knot-spacing 12 --degree 3 \
  --out out/align_spline.nxs

# Low-order polynomial model for simple scan-length drift
uv run tomojax-align --data data/sim_misaligned.nxs \
  --levels 4 2 1 --opt-method gn \
  --pose-model polynomial --degree 2 \
  --out out/align_poly2.nxs

# 5-DOF full alignment is the default; this explicit form is equivalent
uv run tomojax-align --data data/sim_misaligned.nxs \
  --levels 4 2 1 --opt-method gn --optimise-dofs alpha,beta,phi,dx,dz \
  --out out/align_full_5dof.nxs

# TOML config with an explicit CLI override
uv run tomojax-align --config docs/align_config.toml --levels 2 1
```


## convert

Convert between `.npz` and `.nxs` (HDF5). In `.nxs`, data follow the NXtomo convention with TomoJAX extras. See `docs/schema_nxtomo.md`.

Usage
```
python -m tomojax.cli.convert --in <in.npz|in.nxs> --out <out.nxs|out.npz>
```

Examples
```
uv run tomojax-convert --in data/sim_aligned.nxs --out data/sim_aligned.npz
uv run tomojax-convert --in data/sim_aligned.npz --out data/sim_aligned_back.nxs
```


## Environment variables

- `TOMOJAX_PROGRESS=1` — enable progress bars.
- `TOMOJAX_JAX_CACHE_DIR=<path>` — persistent JAX compilation cache directory.
- `JAX_PLATFORM_NAME=cpu` or `JAX_PLATFORMS=cuda` — pin backend selection.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` — disable device memory preallocation.
