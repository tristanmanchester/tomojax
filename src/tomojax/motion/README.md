# tomojax.motion

## Purpose

`tomojax.motion` provides motion-estimation primitives for alignment
initialization and regularization, plus typed object-frame motion traces.

## Public API

- `phase_corr_shift(ref, tgt)`
- `ObjectMotionTrace`
- `read_object_motion_csv(path)`
- `write_object_motion_csv(path, trace)`

## Dependencies

JAX only. Must not depend on alignment, reconstruction, datasets, or CLI.

## Invariants

- Public imports go through `tomojax.motion`, not private `_phasecorr`.
- Phase-correlation shifts use half-open wrapped intervals for even detector
  dimensions.
- Vectorizable with `jax.vmap`.
- Object-motion traces are 1D per-view arrays with matching view counts.

## Tests

Exercised indirectly through simulation and alignment workflow tests.
