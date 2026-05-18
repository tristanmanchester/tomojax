# tomojax.forward

## Purpose

`tomojax.forward` owns differentiable projection, projection-domain residuals,
and geometry-parameter reductions for the v2 core ray path.

The supported v2 projector family is `core_trilinear_ray`: v2 `GeometryState`
is adapted to core `Grid`, `Detector`, detector grids, and per-view 4x4
`T_all`, then projected with `tomojax.core.projector.forward_project_view_T`.
The old rotate-and-sum approximation is not an operational v2 path.

## Public API

- `project_parallel_reference`
- `project_parallel_reference_arrays`
- `core_projection_geometry_from_state`
- `core_projection_geometry_from_arrays`
- `nominal_axis_unit_from_geometry`
- `CoreProjectionGeometry`
- `PROJECTION_OPERATOR`
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

- `core_trilinear_ray` is the single supported v2 operator family.
- Projection residuals must support masks and robust whitening.
- Backend fast paths must report provenance and compare against the reference
  path.
- Supported parallel-tomography DOFs are nominal theta, theta scale, theta
  offset, per-view alpha/beta/phi residuals, detector u/v shift, detector roll,
  axis x/y tilt, and per-view dx/dz.
- Detector roll uses a calibrated detector-grid transform: roll is applied
  around the zero-centre detector plane, while detector centre offsets remain
  independent.
- Axis rotations use the core rotation-axis pose convention: acquisition
  metadata supplies the nominal parallel-tomography or parallel-laminography
  axis, x/y setup rotations apply corrections on top of it, and `T_all` is
  built with `axis_pose_stack`.
- Alpha/beta pose rotations are composed after the nominal axis/theta pose in
  object coordinates, matching the sidecar geometry wrapper convention.
- Parallel laminography is represented as a tilted nominal rotation axis for
  the same `core_trilinear_ray` operator. Object drift is an explicit
  unsupported state until its core convention mapping is defined.
- Residual filters are projection-domain JAX reference policies. The current
  public policies are `raw`, `lowpass_gaussian`, and
  `bandpass_difference_of_gaussians`.

## Tests

- `tests/test_v2_module_skeleton.py` verifies this facade exists and imports.
- `tests/test_forward_reference.py` covers the v2-to-core adapter, supported
  detector/theta shifts, detector roll, axis tilt, alpha/beta pose rotations,
  parallel laminography nominal-axis mapping, and robust residual contracts.
