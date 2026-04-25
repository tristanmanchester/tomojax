# align

The `tomojax-align` command performs joint per-view alignment and
reconstruction using alternating TV reconstruction and pose
optimization. It supports single-level or multi-resolution
coarse-to-fine alignment with multiple optimizer choices.

```
tomojax-align [--config <config.toml>] --data <in.nxs> \
  --out <out.nxs> [options...]
```

## Outer and inner loops

Each outer iteration reconstructs the volume with a TV-regularised
solver, then updates the per-view alignment parameters to reduce a
chosen loss. You control this alternation with these flags.

- `--outer-iters` -- number of reconstruct-then-align cycles
  (default: 5)
- `--recon-iters` -- reconstruction iterations per outer step
  (default: 10)
- `--lambda-tv` -- TV regularisation weight for the inner
  reconstruction (default: 0.005)
- `--recon-algo fista|spdhg` -- inner reconstruction solver
  (default: `fista`). SPDHG uses stochastic view subsets and can
  scale better for large view counts; control its subset size with
  `--views-per-batch` and its deterministic order with
  `--spdhg-seed`.
- `--tv-prox-iters` -- inner TV proximal iterations for FISTA
  (default: 10). We recommend increasing to 20--30 for noisy data.
- `--regulariser tv|huber_tv` -- regulariser for the inner
  reconstruction (default: `tv`). `huber_tv` is smoother near zero
  gradients; tune `--huber-delta` (default: 0.01) to control the
  transition.
- `--recon-positivity` / `--no-recon-positivity` -- positivity
  constraint for SPDHG inner reconstructions (default: on). This
  flag has no effect when `--recon-algo fista` is used.

## Optimizer selection

The alignment step minimizes a similarity loss between the current
reconstruction's forward projections and the measured data. Three
optimizers are available. See [alignment concepts](../concepts/alignment.md)
for background on each approach.

**Gradient descent** (`--opt-method gd`):

- `--lr-rot` -- learning rate for rotations (default: 1e-3)
- `--lr-trans` -- learning rate for translations (default: 1e-1)

**Gauss-Newton** (`--opt-method gn`, default):

- `--gn-damping` -- Levenberg-Marquardt damping factor
  (default: 1e-3)

GN is supported for L2-like losses: `l2`, `l2_otsu`, `edge_l2`,
`pwls`. It's usually the best choice for these losses.

**L-BFGS** (`--opt-method lbfgs`):

- `--lbfgs-maxiter` -- maximum L-BFGS iterations per outer step
  (default: 20)
- `--lbfgs-ftol` -- relative function tolerance (default: 1e-6)
- `--lbfgs-gtol` -- gradient-norm tolerance (default: 1e-5)
- `--lbfgs-maxls` -- maximum line-search steps per iteration
  (default: 20)
- `--lbfgs-memory-size` -- number of stored gradient/step pairs
  (default: 10)

L-BFGS is useful for differentiable robust or similarity losses
where GN isn't available. If a numerical failure occurs, the run
logs the reason and falls back to GD for that step.

## DOF selection and bounds

By default, all five degrees of freedom are optimised: `alpha`,
`beta`, `phi`, `dx`, `dz`. You can restrict or constrain them.

- `--optimise-dofs` -- named DOFs to optimise (e.g., `dx,dz` for
  translation-only alignment)
- `--freeze-dofs` -- named DOFs to keep fixed at initial values
  (e.g., `phi` to freeze in-plane spin)
- `--bounds dx=-20:20,dz=-20:20,alpha=-0.05:0.05` -- finite
  per-DOF parameter bounds. Rotations (`alpha`, `beta`, `phi`) are
  in radians; translations (`dx`, `dz`) are in world units.
  Omitted DOFs are unconstrained. Frozen DOFs stay fixed even if a
  bound is supplied.

> [!NOTE]
> DOF selection doesn't change the saved parameter format. Outputs
> always use five columns in `[alpha, beta, phi, dx, dz]` order,
> with inactive columns held at their initial values.

## Pose models

The default `per_view` model optimises one independent 5-DOF vector
per view. Smooth models reduce the number of free parameters and
are better for drift, stage sag, or thermal trends.

- `--pose-model per_view` -- independent parameters per view
  (default)
- `--pose-model polynomial` -- fit each active DOF as a
  low-degree polynomial over the scan coordinate
- `--pose-model spline` -- optimise smooth knot trajectories and
  expand them to per-view parameters

**Shared model flags:**

- `--knot-spacing` -- view spacing between spline knots
  (default: 8)
- `--degree` -- polynomial degree or spline degree (default: 3)
- `--w-rot` -- smoothness weight for rotations (default: 1e-3)
- `--w-trans` -- smoothness weight for translations (default: 1e-3)

> [!TIP]
> Smooth models usually need less explicit smoothness prior because
> the basis already constrains freedom. Keep `per_view` for abrupt
> shifts, dropped-view artifacts, or genuinely view-local motion.

## Multi-resolution

Coarse-to-fine alignment starts at a downsampled resolution and
refines at successively finer levels. This improves convergence for
large misalignments.

- `--levels 4 2 1` -- pyramid factors from coarsest to finest
- `--seed-translations` -- use phase correlation to initialise
  `dx,dz` at the coarsest level

## Instrument Geometry Blocks

`tomojax-align` can solve static instrument geometry before the
per-view pose update at each pyramid level. This uses the same
coarse-to-fine alignment loop and differentiable projector as pose
alignment, rather than a separate calibration command.

- `--optimise-geometry det_u_px` -- estimate the horizontal
  detector/ray-grid centre offset in native detector pixels.
- `--optimise-geometry detector_roll_deg` -- estimate static detector
  roll in degrees.
- `--optimise-geometry axis_rot_x_deg,axis_rot_y_deg` -- estimate the
  lab-frame rotation-axis direction as small rotations in degrees.
  `tilt_deg` is accepted as a laminography-friendly alias for the
  axis component matching the scan's tilt direction.

Geometry blocks are staged before pose blocks at each multiresolution
level, so common runs should use the normal pyramid, for example
`--levels 8 4 2 1`. To run geometry-only calibration, freeze all pose
DOFs:

```bash
tomojax-align --data data/scan.nxs \
  --levels 8 4 2 1 \
  --optimise-geometry det_u_px,detector_roll_deg \
  --freeze-dofs alpha,beta,phi,dx,dz \
  --out out/geometry_calibrated.nxs
```

## Loss selection

The alignment step minimizes a similarity measure between simulated
and measured projections. You can use a single loss for all levels
or specify a per-level schedule.

- `--loss <name>` -- loss for all levels (default: `l2_otsu`).
  Available losses include `l2`, `charbonnier`, `huber`, `ssim`,
  `ms-ssim`, `zncc`, `mi`, `phasecorr`, `edge_l2`, `ngf`,
  `l2_otsu`, and many more.
- `--loss-schedule 4:phasecorr,2:ssim,1:l2_otsu` -- override the
  loss per pyramid factor. Numeric keys refer to `--levels` values.
  Levels omitted from the schedule fall back to `--loss`.
- `--loss-param k=v` -- repeatable loss parameter, e.g.,
  `delta=1.0`, `eps=1e-3`, `window=7`.

**Loss schedule guidance.** The following recipes have been
validated on small smoke tests across multiple seeds.

For translation-only coarse-to-fine alignment:

```bash
tomojax-align --optimise-dofs dx,dz \
  --loss-schedule 4:phasecorr,2:ssim,1:l2_otsu ...
```

For full 5-DOF alignment, we recommend starting with a conservative
image-similarity loss:

```bash
tomojax-align --loss ssim ...
# or
tomojax-align --loss charbonnier ...
```

Treat `phasecorr` as a coarse translation helper, not a good
all-level loss. Use `l2_otsu` as a fine-level stabiliser or
fallback.

See [loss functions reference](../reference/loss-functions.md) for
the full catalogue and parameter descriptions.

## Gauge fixing

Alignment parameters have an inherent ambiguity in the global
detector translation. Gauge fixing removes this ambiguity so saved
parameters are easier to interpret as residual alignment motion.

- `--gauge-fix mean_translation` -- subtract the scan-wide mean
  from active `dx,dz` after initialisation and pose updates
  (default)
- `--gauge-fix none` -- preserve historical unconstrained traces

See [alignment-gauge-benchmark.md](../internal/alignment-gauge-benchmark.md)
for a 64³ validation comparison.

## Early stopping

Early stopping terminates the outer loop when alignment improvement
stalls, saving time on converged runs.

- `--early-stop` / `--no-early-stop` -- enable or disable early
  stopping (default: on)
- `--early-stop-rel` -- relative improvement threshold
  (default: 1e-3)
- `--early-stop-patience` -- consecutive outers below threshold
  before stopping (default: 2)

## Checkpoint and resume

Long alignment runs can be checkpointed so you can resume after
interruptions. Checkpoints are written at outer-iteration
boundaries only, not mid-FISTA inner-iteration snapshots.

- `--checkpoint PATH` -- write an atomic `.npz` checkpoint after
  completed outer iterations
- `--checkpoint-every N` -- checkpoint every N completed global
  outer iterations (default: 1 when checkpointing is enabled)
- `--resume PATH` -- load a checkpoint and continue from the next
  outer iteration or pyramid level. If `--checkpoint` is omitted,
  future checkpoints are written back to the resume path.

> [!NOTE]
> Checkpoint files include the volume, alignment parameters, loss
> history, and metadata needed for validation. The resume path
> must match the current run configuration.

## Parameter exports

You can export the final per-view alignment parameters as sidecar
files for plotting and reproducibility.

- `--save-params-json PATH` -- write a JSON sidecar
- `--save-params-csv PATH` -- write a CSV sidecar

Both formats include the following columns: `alpha_rad`,
`beta_rad`, `phi_rad`, `dx_world`, `dz_world`, `dx_px`, `dz_px`.
The selected gauge operation is recorded in output metadata and
JSON sidecars.

## Examples

GN multi-resolution:

```bash
tomojax-align --data data/scan_misaligned.nxs \
  --levels 4 2 1 --outer-iters 4 --recon-iters 25 \
  --lambda-tv 0.003 \
  --opt-method gn --gn-damping 1e-3 \
  --gather-dtype bf16 --checkpoint-projector \
  --checkpoint out/align.checkpoint.npz \
  --log-summary --out out/align.nxs \
  --save-params-json out/align.params.json \
  --save-params-csv out/align.params.csv \
  --save-manifest out/align.manifest.json \
  --progress
```

GD single level:

```bash
tomojax-align --data data/scan_misaligned.nxs \
  --outer-iters 6 --recon-iters 30 --lambda-tv 0.005 \
  --opt-method gd --lr-rot 3e-3 --lr-trans 1e-1 \
  --out out/align_gd.nxs --progress
```

L-BFGS with a robust loss:

```bash
tomojax-align --data data/scan_misaligned.nxs \
  --levels 4 2 1 --outer-iters 3 --recon-iters 25 \
  --lambda-tv 0.003 \
  --opt-method lbfgs --lbfgs-maxiter 20 \
  --loss charbonnier \
  --log-summary --out out/align_lbfgs.nxs
```

2-DOF translation-only:

```bash
tomojax-align --data data/scan_misaligned.nxs \
  --levels 4 2 1 --opt-method gn \
  --optimise-dofs dx,dz \
  --out out/align_translation.nxs
```

Bounded translations and one rotation:

```bash
tomojax-align --data data/scan_misaligned.nxs \
  --levels 4 2 1 --opt-method gn \
  --bounds dx=-20:20,dz=-20:20,alpha=-0.05:0.05 \
  --out out/align_bounded.nxs
```

Smooth spline model:

```bash
tomojax-align --data data/scan_misaligned.nxs \
  --levels 4 2 1 --opt-method gn \
  --pose-model spline --knot-spacing 12 --degree 3 \
  --out out/align_spline.nxs
```

Coarse-to-fine loss schedule:

```bash
tomojax-align --data data/scan_misaligned.nxs \
  --levels 4 2 1 \
  --loss-schedule 4:phasecorr,2:ssim,1:l2_otsu \
  --opt-method gn \
  --out out/align_loss_schedule.nxs
```

TOML config with an explicit override:

```bash
tomojax-align --config docs/align_config.toml --levels 2 1
```

Resume a checkpointed run:

```bash
tomojax-align --data data/scan_misaligned.nxs \
  --levels 4 2 1 --outer-iters 4 --recon-iters 25 \
  --lambda-tv 0.003 \
  --opt-method gn --gn-damping 1e-3 \
  --gather-dtype bf16 --checkpoint-projector \
  --resume out/align.checkpoint.npz \
  --out out/align.nxs
```

---

See also: [alignment concepts](../concepts/alignment.md)
| [loss functions](../reference/loss-functions.md)
| [config files](../reference/config-files.md)
| [CLI overview](index.md)
