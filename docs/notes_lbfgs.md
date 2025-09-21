# Notes: Re-introducing L-BFGS for Alignment

## Background

- Historical commits (pre-v2) such as `f8b38f50` used Optax’s `lbfgs` for pose
  refinement alongside AdaBelief, Adam, NAdam, YOGI, and GD.
- The code lived in `alignment-testing/optimization_steps.py` and wrapped the
  alignment loss in a flattened `(n_views * 5,)` parameter vector.
- Migration to the current `src/tomojax/align/pipeline.py` removed those scripts,
  leaving only gradient descent and Gauss–Newton.

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

## Porting to the Current Pipeline

- We already have `align_loss(params5, x)` and its gradient.
- Steps to integrate:
  1. Flatten `params5` → `(n_views * 5,)` before calling Optax.
  2. Build `loss_fn(flat)` that reshapes, calls `align_loss_jit`, and returns
     both loss and grads (`jax.value_and_grad`).
  3. Instantiate `optax.lbfgs(...)` (consider zoom line search).
  4. Run a small fixed number of L-BFGS iterations per outer loop or until
     convergence; reshape back to `(n_views, 5)` afterward.
  5. Re-use the rotation/translation scaling if needed or tune per loss.

## Caveats & TODOs

- L-BFGS is general, so it should handle all the new loss functions, but line
  search tolerances may need tuning for MI/SSIM/phase correlation.
- Memory grows with `(history_size * n_views * 5)`, so we should gate it behind
  `cfg.views_per_batch` or allow down-sampling for large scan counts.
- Gauss–Newton remains the best choice for pure L2; we can expose `opt_method`
  values `{gd, gn, lbfgs}` and keep sensible defaults per loss.
- When we resume loss work: re-run the smoke tests with `opt_method=lbfgs` once
  it is wired in and ensure the line search respects early stopping flags.

