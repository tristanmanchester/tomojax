# tomojax.forward

## Purpose

`tomojax.forward` provides differentiable projection, projection-domain
residuals, and geometry-parameter reductions.

The projector adapts `GeometryState` to core `Grid`, `Detector`, detector
grids, and per-view 4x4 `T_all`, then projects with
`tomojax.core.projector.forward_project_view_T`.

## Public API

- `project_parallel_reference`
- `project_parallel_reference_from_input`
- `ProjectionArrayGeometryInput`
- `ProjectionOperatorName`
- `core_projection_geometry_from_state`
- `core_projection_geometry_from_input`
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

Allowed: `tomojax.core`, `tomojax.geometry`, `tomojax.motion`,
`tomojax.nuisance`, `tomojax.backends`.

Forbidden: private files from other modules, reconstruction/alignment
orchestration, Pallas fast paths without JAX-reference equivalence tests.

## Invariants

- Projection residuals support masks and robust whitening.
- Backend fast paths report provenance and compare against the reference path.
- Supported DOFs: nominal theta, theta scale/offset, per-view alpha/beta/phi
  residuals, detector u/v shift, detector roll, axis x/y tilt, per-view dx/dz.
- Detector roll applies around the zero-centre detector plane; detector centre
  offsets are independent.
- Axis rotations use the core rotation-axis pose convention: nominal axis from
  acquisition metadata, x/y setup corrections on top, `T_all` built with
  `axis_pose_stack`.
- Alpha/beta pose rotations compose after nominal axis/theta in object
  coordinates.
- Parallel laminography uses a tilted nominal rotation axis with the same
  projector.
- Residual filter policies: `raw`, `lowpass_gaussian`,
  `bandpass_difference_of_gaussians`.

## Tests

- `tests/test_numerical_engines.py` verifies the public grouped-input projection
  path and core numerical invariants used by reconstruction.
- `tests/test_product_surface.py` verifies the public module import surface.
