# tomojax.motion

## Purpose

`tomojax.motion` owns motion-estimation primitives used to initialize or
regularize alignment. The current Milestone 0 surface is the phase-correlation
translation estimator migrated out of the forbidden `tomojax.utils` namespace.

## Public API

- `phase_corr_shift(ref, tgt)`

## Dependencies

This module may depend on JAX numerical primitives. It must not depend on
alignment orchestration, reconstruction solvers, datasets, or CLI modules.

## Invariants

- Public imports go through `tomojax.motion`, not private `_phasecorr`.
- Phase-correlation shifts use the half-open wrapped interval convention for
  even detector dimensions.
- The implementation remains vectorizable with `jax.vmap`.

## Tests

Covered by `tests/test_phasecorr.py`.
