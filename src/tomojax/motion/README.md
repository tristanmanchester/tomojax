# tomojax.motion

## Purpose

`tomojax.motion` owns motion-estimation primitives used to initialize or
regularize alignment, plus typed object-frame motion traces used by synthetic
benchmarks and future object-motion solvers.

## Public API

- `phase_corr_shift(ref, tgt)`
- `ObjectMotionTrace`
- `read_object_motion_csv(path)`
- `write_object_motion_csv(path, trace)`

## Dependencies

This module may depend on JAX numerical primitives. It must not depend on
alignment orchestration, reconstruction solvers, datasets, or CLI modules.

## Invariants

- Public imports go through `tomojax.motion`, not private `_phasecorr`.
- Phase-correlation shifts use the half-open wrapped interval convention for
  even detector dimensions.
- The implementation remains vectorizable with `jax.vmap`.
- Object-motion traces are one-dimensional per-view arrays with matching view
  counts.

## Tests

Not yet covered by dedicated product tests. Phase-correlation and object-motion
contracts are exercised indirectly through simulation and alignment workflows.
