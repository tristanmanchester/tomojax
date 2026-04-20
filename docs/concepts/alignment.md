# Alignment

TomoJAX jointly estimates per-view alignment parameters and a
reconstructed volume. This page explains the algorithm, the available
optimizers, and the key tuning knobs. For CLI flags and examples, see
the [align CLI reference](../cli/align.md).

## The alignment problem

In real CT scans, mechanical drift, thermal expansion, and vibration
cause each projection view to deviate from its nominal pose. Naive
reconstruction with wrong poses produces blurred or streaked volumes.
TomoJAX solves this by optimizing 5-DOF rigid-body parameters per
view (see [Geometry](geometry.md)) so that reprojected data matches
the measured projections.

## Alternating optimization

Each outer iteration performs two steps:

1. **Reconstruct** — fix the current alignment parameters and run
   FISTA-TV (or SPDHG-TV) to update the volume.
2. **Align** — fix the reconstructed volume and optimize per-view pose
   parameters to minimize the projection residual.

This alternation repeats for `--outer-iters` iterations (default 5).
Because the projector is fully differentiable, gradients flow through
the trilinear interpolation into the pose parameters.

## Multi-resolution pyramid

Starting at full resolution risks converging to a local minimum.
TomoJAX supports coarse-to-fine optimization with `--levels`:

```bash
--levels 4 2 1
```

This downsamples projections by 4x, then 2x, then runs at full
resolution. At each level:

1. A new reconstruction and alignment run executes for the configured
   number of outer iterations.
2. Parameters are transferred to the next finer level and rescaled
   (translations are multiplied by the level ratio).

Coarse levels resolve large misalignments quickly. Fine levels refine
sub-pixel accuracy.

## Optimizers

TomoJAX provides three optimizers for the alignment step. Choose based
on your loss function and data quality.

### Gradient descent (GD)

The simplest option. Set learning rates with `--lr-rot` and
`--lr-trans`:

```bash
--opt-method gd --lr-rot 3e-3 --lr-trans 1e-1
```

GD works but typically needs more outer iterations than GN.

### Gauss-Newton (GN)

The recommended default for L2-like losses. GN uses a second-order
approximation with Levenberg-Marquardt damping:

```bash
--opt-method gn --gn-damping 1e-3
```

GN converges faster than GD and handles the curvature of the L2
objective well. It rejects steps that increase the loss.

### L-BFGS

An Optax-based limited-memory quasi-Newton optimizer, best for
differentiable robust or similarity losses (for example, Charbonnier,
SSIM) where the GN approximation isn't available:

```bash
--opt-method lbfgs --lbfgs-maxiter 20 --loss charbonnier
```

L-BFGS optimizes only the pose parameters (not the volume). If a
numerical failure occurs, the run falls back to GD for that step and
logs the reason. Use `--log-summary` to see accepted/rejected status.

## Pose models

The pose model controls how per-view parameters are represented.

### Per-view (default)

Each view has an independent 5-DOF parameter vector. This is best for
abrupt shifts, dropped views, or genuinely view-local motion:

```bash
--pose-model per_view
```

### Spline

Fits smooth B-spline trajectories through knot parameters and expands
them to per-view values. Best for slow drift, stage sag, or thermal
trends:

```bash
--pose-model spline --knot-spacing 12 --degree 3
```

### Polynomial

Fits each active DOF as a low-degree polynomial over the scan
coordinate. Best for simple scan-length trends:

```bash
--pose-model polynomial --degree 2
```

> [!TIP]
> Smooth models reduce overfitting on noisy data by constraining
> the parameter space. They also need less explicit smoothness
> regularization (`--w-rot`, `--w-trans`).

## Active DOFs and bounds

By default, all five DOFs (`alpha`, `beta`, `phi`, `dx`, `dz`) are
optimized. You can restrict or freeze specific DOFs.

**Translation-only alignment (2-DOF):**

```bash
--optimise-dofs dx,dz
```

**Freeze in-plane rotation (4-DOF):**

```bash
--freeze-dofs phi
```

**Bound active DOFs to physical limits:**

```bash
--bounds dx=-20:20,dz=-20:20,alpha=-0.05:0.05
```

Rotations are in radians and translations are in world units. Frozen
DOFs stay fixed even if bounds are supplied for them. Outputs always
store all five columns in `[alpha, beta, phi, dx, dz]` order;
inactive columns hold their initial values.

## Gauge fixing

In parallel-beam CT, a global detector translation is ambiguous — the
same projection data can be explained by shifting every view by the
same offset. Without gauge fixing, optimization can drift the
absolute translation values while the relative alignment stays
correct.

TomoJAX applies gauge fixing by default:

```bash
--gauge-fix mean_translation
```

This subtracts the scan-wide mean from active `dx` and `dz` after
initialization and after each pose update. The saved parameters
represent per-view residual translations, which are easier to
interpret.

Use `--gauge-fix none` to preserve historical unconstrained traces.

## Early stopping

Alignment stops early if the objective improvement plateaus:

- `--early-stop-rel` (default `1e-3`) — minimum relative improvement
  per outer iteration
- `--early-stop-patience` (default `2`) — consecutive iterations
  below the threshold before stopping

Use `--log-summary` to see when early stopping triggers. Disable with
`--no-early-stop`.

## Loss functions

The alignment loss measures how well reprojected data matches the
measured projections. TomoJAX provides multiple losses with different
trade-offs. See the [Loss functions reference](../reference/loss-functions.md)
for the full list and scheduling guide.

## Next steps

- [align CLI reference](../cli/align.md) — all flags, examples, and
  TOML config
- [Loss functions](../reference/loss-functions.md) — available losses
  and scheduling
- [Reconstruction concepts](reconstruction.md) — the inner
  reconstruction step
