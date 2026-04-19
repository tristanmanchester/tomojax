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

Reconstruct a volume from projections via FBP, FISTA, or SPDHG. Saves a new `.nxs` with the reconstruction.

Usage
```
python -m tomojax.cli.recon [--config <config.toml>] --data <in.nxs> \
  [--algo fbp|fista|spdhg] [--filter ramp|shepp|hann] \
  [--iters <int>] [--lambda-tv <float>] [--regulariser tv|huber_tv] [--huber-delta <float>] \
  [--tv-prox-iters <int>] [--L <float>] \
  [--positivity|--no-positivity] [--lower-bound <float>] [--upper-bound <float>] \
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
- FISTA: `--iters` (50), `--lambda-tv` (0.005), `--tv-prox-iters` (10) controls the inner TV proximal iterations (use 20–30 for heavy noise), optional fixed `--L` to skip power‑method. Add `--positivity`, `--lower-bound`, and/or `--upper-bound` to project each iterate onto common physical voxel constraints. See `docs/fista_constraints_validation_64.md` for the `64^3` constraint validation smoke test.
- Regulariser: `--regulariser tv` is the default and uses the existing isotropic TV path. `--regulariser huber_tv` uses a smoother Huber-TV penalty; smaller `--huber-delta` is more TV/L1-like, while gradients below `huber_delta` are penalised quadratically.
- SPDHG: `--iters` (outer PDHG steps), `--lambda-tv` (regularisation weight), `--views-per-batch` (stochastic block size, e.g. 16–64), `--theta` (extrapolation, e.g. 0.5–1.0). Step sizes default to operator‑norm‑based auto; override with `--spdhg-tau`, `--spdhg-sigma-data`, `--spdhg-sigma-tv`. Use `--spdhg-seed` to fix block order.
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

# FISTA with positivity and box constraints
uv run tomojax-recon --data data/sim_misaligned.nxs \
  --algo fista --iters 80 --lambda-tv 0.005 \
  --positivity --lower-bound 0 --upper-bound 1 \
  --out out/fista_bounded.nxs --progress

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

Joint per‑view alignment and reconstruction (alternating TV reconstruction and alignment updates). Supports single‑level or multi‑resolution.

Usage
```
python -m tomojax.cli.align [--config <config.toml>] --data <in.nxs> \
  [--outer-iters <int>] [--recon-iters <int>] [--lambda-tv <float>] \
  [--regulariser tv|huber_tv] [--huber-delta <float>] \
  [--recon-algo fista|spdhg] [--tv-prox-iters <int>] \
  [--views-per-batch <int>] [--spdhg-seed <int>] \
  [--recon-positivity|--no-recon-positivity] \
  [--lr-rot <float>] [--lr-trans <float>] \
  [--opt-method gd|gn|lbfgs] [--gn-damping <float>] \
  [--lbfgs-maxiter <int>] [--lbfgs-ftol <float>] [--lbfgs-gtol <float>] \
  [--lbfgs-maxls <int>] [--lbfgs-memory-size <int>] \
  [--optimise-dofs <names>] [--freeze-dofs <names>] [--bounds <spec>] \
  [--early-stop|--no-early-stop] [--early-stop-rel <float>] [--early-stop-patience <int>] \
  [--pose-model per_view|polynomial|spline] [--knot-spacing <int>] [--degree <int>] \
  [--gauge-fix mean_translation|none] \
  [--w-rot <float>] [--w-trans <float>] [--seed-translations] \
  [--loss <name>] [--loss-schedule <LEVEL:LOSS,...>] \
  [--levels <ints...>] [--gather-dtype fp32|bf16|fp16] \
  [--checkpoint-projector|--no-checkpoint-projector] [--recon-L <float>] [--log-summary] \
  [--checkpoint <checkpoint.npz>] [--checkpoint-every <N>] [--resume <checkpoint.npz>] \
  --out <out.nxs> [--save-params-json <out.json>] [--save-params-csv <out.csv>] \
  [--save-manifest <manifest.json>] [--progress]
```

Key options
- Config: `--config docs/align_config.toml` loads flat TOML defaults. Explicit CLI flags override file values, including list values such as `--levels 2 1` and booleans such as `--no-checkpoint-projector`.
- Outer/inner loops: `--outer-iters` (5), `--recon-iters` (10), `--lambda-tv` (0.005), `--recon-algo fista|spdhg` (`fista` default), `--tv-prox-iters` (10; FISTA only, increase to 20–30 for noisy data).
- Regulariser: `--regulariser tv` keeps the standard inner reconstruction TV behavior. `--regulariser huber_tv` is smoother near zero gradients; tune `--huber-delta` to control the transition from quadratic smoothing to TV-like edge behavior.
- Inner solver choice: FISTA is the conservative default for continuity with existing workflows. SPDHG uses stochastic view subsets and can scale better for larger view counts; control its subset size with `--views-per-batch` and deterministic subset order with `--spdhg-seed`.
- SPDHG details: `--recon-positivity` is enabled by default and applies only to SPDHG. SPDHG objective logs are minibatch estimates, so compare their trend rather than treating them as identical to FISTA's full objective trace.
- Smoke benchmark: see `docs/alignment_inner_solver_benchmark_64.md` for a `64^3` CPU comparison of FISTA and SPDHG as alignment inner solvers.
- Alignment step: gradient descent (`--lr-rot`, `--lr-trans`), Gauss‑Newton (`--opt-method gn`, `--gn-damping`), or Optax L‑BFGS (`--opt-method lbfgs`, `--lbfgs-maxiter`, `--lbfgs-memory-size`). GN is usually best for L2-like losses; L‑BFGS is useful for differentiable robust or similarity losses where GN is not available. See `docs/alignment_lbfgs_benchmark_64.md` for `64^3` CPU comparison and tuning notes.
- Active DOFs: choose from `alpha`, `beta`, `phi`, `dx`, `dz`. By default all five are optimised. Use `--optimise-dofs dx,dz` for translation-only alignment, or `--freeze-dofs phi` to keep selected parameters fixed at their initial values.
- Parameter bounds: `--bounds dx=-20:20,dz=-20:20,alpha=-0.05:0.05` clips named DOFs after each update. Rotations (`alpha`, `beta`, `phi`) are in radians; translations (`dx`, `dz`) are in world units. Omitted DOFs are unconstrained, and frozen DOFs stay fixed even if a bound is supplied for them. L‑BFGS optimises an unconstrained active-DOF vector and maps it through the same bounds before evaluating the JAX objective; smooth pose models project expanded per-view parameters back into those bounds.
- Pose model: `--pose-model per_view` (default) optimises one independent parameter vector per view. `--pose-model spline --knot-spacing N --degree 3` optimises smooth knot trajectories and expands them back to per-view parameters before projection. `--pose-model polynomial --degree D` fits each active DOF as a low-degree polynomial over the scan coordinate.
- Gauge fixing: `--gauge-fix mean_translation` is the default and subtracts the
  scan-wide mean from active `dx,dz` after initialization and pose updates. This
  removes the ambiguous global detector translation from per-view traces, so
  saved parameters are easier to interpret as residual alignment motion. Use
  `--gauge-fix none` to reproduce historical unconstrained traces. See
  `docs/alignment_gauge_benchmark_64.md` for a `64^3` validation comparison.
- Early stopping (alignment across outers):
  - Enable/disable: `--early-stop` (default) or `--no-early-stop`.
  - Threshold and patience: `--early-stop-rel` (default 1e-3), `--early-stop-patience` (default 2).
- Smoothness across views: `--w-rot`, `--w-trans` (default 1e‑3 in CLI) add a second-difference prior on the expanded per-view parameters. Smooth pose models usually need less explicit prior strength because the basis already reduces freedom.
- Multi‑resolution: `--levels 4 2 1` for coarse→fine; optional `--seed-translations` uses phase correlation at the coarsest level.
- Loss scheduling: `--loss` selects one loss for every level. `--loss-schedule 4:phasecorr,2:ssim,1:l2_otsu` overrides the loss per pyramid factor, so numeric keys refer to `--levels` values. Levels omitted from the schedule fall back to `--loss`.
- Memory/performance: same knobs as recon (gather dtype and checkpointing).
- `--recon-L`: FISTA-only fixed Lipschitz constant to skip per‑level power‑method if you already know a good bound.
- Checkpoint/resume: `--checkpoint PATH` writes an atomic `.npz` checkpoint after completed alignment outer iterations, and `--checkpoint-every N` controls the completed global outer-iteration cadence. `--resume PATH` loads a checkpoint and continues from the next outer iteration or pyramid level; if `--checkpoint` is omitted, future checkpoints are written back to the resume path. Checkpoints are outer-iteration boundaries only, not mid-FISTA inner-iteration snapshots.
- Parameter exports: `--save-params-json` and `--save-params-csv` write named per-view sidecars with `alpha_rad`, `beta_rad`, `phi_rad`, `dx_world`, `dz_world`, `dx_px`, and `dz_px`.
- Reproducibility: `--save-manifest manifest.json` writes a JSON sidecar with raw argv, parsed CLI args, resolved config, TomoJAX/Python/JAX versions, JAX backend/devices, and a UTC timestamp.
- Early stopping and GN acceptance are enabled by default during alignment: outers stop when relative improvement is tiny, and GN steps are rejected if they don’t reduce loss.
  Use `--log-summary` to see when early stopping triggers.

Notes
- The `.nxs` output still stores the final alignment parameters; JSON/CSV sidecars are optional convenience exports for plotting and reproducibility. The selected gauge operation is recorded in output metadata and JSON sidecars.
- DOF selection does not change the saved parameter format: outputs still use five columns in `[alpha, beta, phi, dx, dz]` order, with inactive columns held fixed.
- Gauge fixing is a runtime constraint on the saved alignment parameters. The
  benchmark loss helpers also expose gauge-aware metrics such as relative-motion
  and gauge-fixed RMSE, but those are scoring tools against known truth rather
  than optimizer constraints.
- Smooth motion models are best for slow drift, stage sag, thermal/mechanical trends, or noisy data where independent per-view parameters overfit. Keep `per_view` for abrupt shifts, dropped-view artifacts, or genuinely view-local motion.
- Optax L‑BFGS optimises only pose/alignment parameters for each outer step, not the reconstruction volume. Bounds are enforced by a differentiable transform, not by a Fortran-style active-set L-BFGS-B backend. If its initial objective/gradient is incompatible with the selected loss or a numerical failure occurs, the run logs the reason and falls back to GD for that step. Use `--log-summary` to see accepted/rejected status, objective values, and iteration counts.
- The align CLI initializes a persistent JAX compilation cache automatically. Set `TOMOJAX_JAX_CACHE_DIR` to control the cache location.

Loss schedule guidance
- For translation-only coarse-to-fine alignment, start with:
  `--optimise-dofs dx,dz --loss-schedule 4:phasecorr,2:ssim,1:l2_otsu`.
  Small `40^3` smoke tests over three seeds reduced translation RMSE most consistently with this schedule. The next-best tested schedule was `4:ssim,2:l2,1:l2_otsu`.
- For full 5-DOF alignment, prefer a conservative image-similarity loss first, especially `--loss ssim` or `--loss charbonnier`. In the same smoke tests, coarse GN-compatible L2-style losses (`l2`, `l2_otsu`, `edge_l2`) often improved translation but over-rotated the poses.
- Treat `phasecorr` as a coarse translation helper, not a good all-level loss. It was useful in the `dx,dz` schedule above, but poor when used at every level.
- Use `l2_otsu` as a fine-level stabilizer or fallback. It is conservative and can reject unsafe GN steps, but it may not move much by itself on coarse levels.

Examples
```
# GN, multires
uv run tomojax-align --data data/sim_misaligned.nxs \
  --levels 4 2 1 --outer-iters 4 --recon-iters 25 --lambda-tv 0.003 \
  --opt-method gn --gn-damping 1e-3 \
  --gather-dtype bf16 --checkpoint-projector \
  --checkpoint out/align_misaligned.checkpoint.npz --checkpoint-every 1 \
  --log-summary --out out/align_misaligned.nxs \
  --save-params-json out/align_misaligned.params.json \
  --save-params-csv out/align_misaligned.params.csv \
  --save-manifest out/align_misaligned.manifest.json \
  --progress

# GN, multires with SPDHG-TV inner reconstructions
uv run tomojax-align --data data/sim_misaligned.nxs \
  --levels 4 2 1 --outer-iters 4 --recon-iters 80 --lambda-tv 0.003 \
  --recon-algo spdhg --views-per-batch 16 --spdhg-seed 0 --recon-positivity \
  --opt-method gn --gn-damping 1e-3 \
  --gather-dtype bf16 --checkpoint-projector \
  --log-summary --out out/align_misaligned_spdhg.nxs \
  --progress

# GD with learning rates (single level)
uv run tomojax-align --data data/sim_misaligned.nxs \
  --outer-iters 6 --recon-iters 30 --lambda-tv 0.005 \
  --opt-method gd --lr-rot 3e-3 --lr-trans 1e-1 \
  --out out/align_gd.nxs --progress

# Optax L-BFGS pose refinement for a differentiable robust loss
uv run tomojax-align --data data/sim_misaligned.nxs \
  --levels 4 2 1 --outer-iters 3 --recon-iters 25 --lambda-tv 0.003 \
  --opt-method lbfgs --lbfgs-maxiter 20 --loss charbonnier \
  --log-summary --out out/align_lbfgs.nxs

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

# Coarse-to-fine loss schedule: robust/coarse objectives early, sharper loss at full resolution
uv run tomojax-align --data data/sim_misaligned.nxs \
  --levels 4 2 1 \
  --loss-schedule 4:phasecorr,2:ssim,1:l2_otsu \
  --opt-method gn \
  --out out/align_loss_schedule.nxs

# Low-order polynomial model for simple scan-length drift
uv run tomojax-align --data data/sim_misaligned.nxs \
  --levels 4 2 1 --opt-method gn \
  --pose-model polynomial --degree 2 \
  --out out/align_poly2.nxs

# 5-DOF full alignment is the default; this explicit form is equivalent
uv run tomojax-align --data data/sim_misaligned.nxs \
  --levels 4 2 1 --opt-method gn --optimise-dofs alpha,beta,phi,dx,dz \
  --out out/align_full_5dof.nxs

# Resume a checkpointed long run
uv run tomojax-align --data data/sim_misaligned.nxs \
  --levels 4 2 1 --outer-iters 4 --recon-iters 25 --lambda-tv 0.003 \
  --opt-method gn --gn-damping 1e-3 \
  --gather-dtype bf16 --checkpoint-projector \
  --resume out/align_misaligned.checkpoint.npz \
  --out out/align_misaligned.nxs

# TOML config with an explicit CLI override
uv run tomojax-align --config docs/align_config.toml --levels 2 1
```


## preprocess

Preprocess raw NXtomo sample/flat/dark frames into sample-only corrected projections.
The command uses `image_key` values `0=sample`, `1=flat`, and `2=dark`.
By default it writes normalised transmission. Add `--log` to write absorption
(`-log` transmission) for reconstruction workflows that expect line integrals.

Usage
```
python -m tomojax.cli.preprocess <raw.nxs> <corrected.nxs> \
  [--log] [--epsilon 1e-6] [--clip-min 1e-6] \
  [--select-views SPEC] [--reject-views SPEC] \
  [--select-views-file PATH] [--reject-views-file PATH] \
  [--auto-reject off|nonfinite|outliers|both] [--outlier-z-threshold 6] \
  [--crop y0:y1,x0:x1]
```

View filtering
- View indices are sample-view indices after `image_key == 0` filtering, not raw
  acquisition frame numbers.
- Specs accept comma or whitespace separated integers and half-open ranges such as
  `0:90` or `120:180:2`. Files use the same syntax and may include `#` comments.
- Selection is applied first, then rejection. Output order always follows the
  original sample-view order.
- `--auto-reject nonfinite` drops corrected views containing NaN/Inf before
  non-finite repair. `--auto-reject outliers` drops robust-MAD outliers by
  per-view median intensity. `both` applies both checks.

Detector crop
- `--crop y0:y1,x0:x1` crops projection axes `(views, y, x)` before correction.
- The output detector metadata is updated to the cropped `nv`/`nu`, and
  `det_center` is shifted so cropped pixel coordinates remain physically aligned
  with the original detector.

Examples
```
uv run tomojax-preprocess raw.nxs corrected_transmission.nxs

uv run tomojax-preprocess raw.nxs corrected_absorption.nxs \
  --log --epsilon 1e-6 --clip-min 1e-6

# Missing flats/darks fail by default; this explicitly uses a zero dark field.
uv run tomojax-preprocess raw_without_darks.nxs corrected_absorption.nxs \
  --log --assume-dark-field 0

# Crop detector ROI before correction.
uv run tomojax-preprocess raw.nxs corrected_cropped.nxs \
  --log --crop 120:900,64:960

# Reject known bad sample views by index/range.
uv run tomojax-preprocess raw.nxs corrected_rejected.nxs \
  --reject-views 12,57:61

# Combine a file of bad views with automatic non-finite/outlier rejection.
uv run tomojax-preprocess raw.nxs corrected_robust.nxs \
  --reject-views-file bad_views.txt --auto-reject both --outlier-z-threshold 6
```

Path overrides
```
uv run tomojax-preprocess raw.nxs corrected.nxs \
  --data-path /entry/imaging/data \
  --angles-path /entry/imaging_sum/smaract_zrot_value_set \
  --image-key-path /entry/instrument/EtherCAT/image_key
```

Notes
- Output `image_key` values are all `0` because the corrected file contains sample
  projections only.
- Provenance is written to `/entry/processing/tomojax/preprocess`, including the source
  dataset paths, frame counts, view filtering, crop bounds, angular coverage before/after
  filtering, correction options, warning counts, and mean flat/dark fields.
- Stripe and ring correction are intentionally not part of this first conservative path.


## inspect

Inspect an NXtomo/HDF5 file before reconstruction. The command reads metadata and projection
statistics only; it does not run JAX or reconstruct a volume.

Usage
```
python -m tomojax.cli.inspect <scan.nxs> [--json <report.json>] [--quicklook <preview.png>]
```

Examples
```
uv run tomojax-inspect data/sim_aligned.nxs
uv run tomojax-inspect data/sim_aligned.nxs \
  --json runs/sim_aligned.inspect.json \
  --quicklook runs/sim_aligned.projection.png
```

Sample output
```
TomoJAX inspection: data/sim_aligned.nxs
Projection shape: [180, 256, 256]
Dtype: float32
Views: 180
Detector shape: {'nv': 256, 'nu': 256}
Stats: min=0, p01=0, mean=0.12, p50=0.08, p99=1.7, max=2.3
NaN/Inf counts: nan=0, +inf=0, -inf=0, inf_total=0
Angle coverage: 179 deg (min=0, max=179, count=180, units=degree)
Geometry type: parallel
Geometry metadata: not found
Detector metadata: nu=256, nv=256, du=1.0, dv=1.0, det_center=[0.0, 0.0]
Flats/darks: not found (image_key present; no flat/dark frames)
Alignment parameters: not found
Memory estimates: grid=[256, 256, 256], fbp_fp32=181403648 bytes, fista_tv_fp32=382730240 bytes, spdhg_tv_fp32=382730240 bytes
```

Notes
- Missing optional metadata is reported as `not found`; valid projection data are enough for a
  successful inspection.
- `--json` writes stable machine-readable keys for automated checks and does not suppress stdout.
- `--quicklook` writes the central projection view as a percentile-scaled PNG.


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
