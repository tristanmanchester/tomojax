# tomojax.forward

## Purpose

`tomojax.forward` owns differentiable projection, projection-domain residuals,
and geometry-parameter reductions for the v2 JAX reference path.

The current implementation is a minimal reference slice for tiny smoke tests:
it supports a simple parallel-beam projection of cubic volumes plus per-view
detector shifts and differentiable in-plane view-angle rotation. Full physical
ray geometry, laminography, detector roll, axis rotations, and Jacobian checks
remain future Phase 2 work.

## Public API

- `project_parallel_reference`
- `project_parallel_reference_arrays`
- `apply_residual_filter`
- `apply_residual_filter_schedule`
- `ResidualFilterConfig`
- `ResidualFilterKind`
- `ResidualFilterResult`
- `masked_whitened_residual`
- `pseudo_huber_loss`
- `pseudo_huber_weights`
- `residual_loss`
- `ResidualResult`

## Dependencies

Allowed future dependencies:

- `tomojax.core`
- `tomojax.geometry`
- `tomojax.motion`
- `tomojax.nuisance`
- `tomojax.backends`

Forbidden dependencies:

- private implementation files from other deep modules
- reconstruction or alignment solver orchestration
- Pallas fast paths as default behavior without JAX-reference equivalence tests

## Invariants

- The JAX reference implementation is the correctness oracle.
- Projection residuals must support masks and robust whitening.
- Backend fast paths must report provenance and compare against the reference
  path.
- Detector shifts in the minimal reference projector use differentiable periodic
  linear interpolation.
- View-angle rotation in the minimal reference projector uses bilinear sampling
  in the x-y plane and zero outside-volume boundaries.
- Residual filters are projection-domain JAX reference policies. The current
  public policies are `raw`, `lowpass_gaussian`, and
  `bandpass_difference_of_gaussians`.

## Tests

- `tests/test_v2_module_skeleton.py` verifies this facade exists and imports.
- `tests/test_forward_reference.py` covers the minimal projector and robust
  residual contracts.
