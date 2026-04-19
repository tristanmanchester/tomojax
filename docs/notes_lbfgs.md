# Notes: Re-introducing L-BFGS for Alignment

## Background

- Historical commits (pre-v2) such as `f8b38f50` used Optax’s `lbfgs` for pose
  refinement alongside AdaBelief, Adam, NAdam, YOGI, and GD.
- The code lived in `alignment-testing/optimization_steps.py` and wrapped the
  alignment loss in a flattened `(n_views * 5,)` parameter vector.
- Migration to the current `src/tomojax/align/pipeline.py` removed those scripts,
  temporarily leaving only gradient descent and Gauss–Newton.

## What the Old Implementation Did

1. **Loss closure** – `compute_loss(p_flat)` converted the flattened params into
   `(n_views, 5)`, injected fixed angles if φ was frozen, projected every view,
   and returned the scalar loss plus its gradient via `jax.value_and_grad`.
2. **Optax L-BFGS** – `optax.lbfgs` with memory size 20 and zoom line search
   produced updates; gradients were re-scaled so rotations (α, β, φ) and
   translations (dx, dz) used different sensitivities.
3. **Iter loop** – Each iteration applied the L-BFGS update, reshaped back to
   `(n_views, 5)`, logged RMS stats, and stopped early when the objective’s
   relative change fell below 1e-6.

## Current v2 Implementation

- `--opt-method lbfgs` uses Optax L-BFGS with zoom line search. The expensive
  objective and gradients stay in JAX; Python only drives the small outer
  optimiser loop and logging.
- The implementation lives in `src/tomojax/align/optimizers.py`; `pipeline.py`
  builds the reconstruction/alignment closures and delegates the pose step.
- The optimiser runs inside each alignment outer step and updates only active
  pose/alignment variables; it does not optimise the reconstruction volume.
- DOF masks are handled by flattening only active per-view pose variables.
  Bounds are handled by a differentiable map from an unconstrained optimiser
  vector into the existing per-DOF lower/upper limits before evaluating the
  objective. Smooth polynomial/spline pose models then project the expanded
  per-view parameters back through the same mask and bounds.
- Per-outer stats and `--log-summary` include objective values, iteration/eval
  counts, line-search steps, selected best/last candidate, acceptance/rejection,
  and fallback reasons.
- `--lbfgs-memory-size` controls the number of previous gradient/step pairs kept
  by Optax's limited-memory inverse-Hessian approximation.
- See `docs/alignment_lbfgs_benchmark_64.md` for the CPU `64^3` smoke-test
  comparisons that motivated the current tuning guidance.

## Caveats & TODOs

- L-BFGS is general, so it can work with many differentiable losses, but line
  search tolerances may need tuning for MI/SSIM/phase correlation.
- Memory grows with `(history_size * n_active_pose_variables)`, so very large
  scans may still prefer GN/GD or a smaller active DOF set.
- The bounded transform is not the original Fortran L-BFGS-B active-set method;
  it keeps TOMOJAX in JAX/Optax while satisfying the same per-DOF bounds at
  objective evaluation time.
- Gauss–Newton remains the best choice for pure L2-like losses.
- If the initial L-BFGS objective/gradient is incompatible with the selected loss
  or a numerical failure occurs, the alignment step logs the reason and falls
  back to GD.
