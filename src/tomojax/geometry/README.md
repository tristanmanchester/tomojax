# tomojax.geometry

## Purpose

`tomojax.geometry` owns public geometry metadata, v2 setup/pose state, gauge
canonicalisation, and detector field-of-view helpers that are shared by
datasets, CLI entrypoints, reconstruction, and alignment.

## Public API

- Axis order constants: `INTERNAL_VOLUME_AXES`, `DISK_VOLUME_AXES`,
  `VOLUME_AXES_ATTR`
- Axis helpers: `axes_to_perm`, `transpose_volume`, `infer_disk_axes`,
  `is_shape_xyz`, `is_shape_zyx`
- FOV helpers: `RoiInfo`, `compute_roi`, `grid_from_detector_fov`,
  `grid_from_detector_fov_cube`, `grid_from_detector_fov_slices`,
  `cylindrical_mask_xy`
- State types: `ScalarParameter`, `SetupParameters`, `PoseParameters`,
  `AcquisitionParameters`, `GeometryState`
- Gauge helpers: `canonicalize_geometry_gauges`, `CanonicalizedGeometry`,
  `GaugeReport`, `GaugeTransfer`
- Artifact helpers: `geometry_state_to_dict`, `geometry_state_from_dict`,
  `write_geometry_json`, `read_geometry_json`, `write_pose_params_csv`,
  `read_pose_params_csv`, `write_pose_decomposition_csv`

## Dependencies

This module currently depends on `tomojax.core.geometry.base` for the existing
`Grid` and `Detector` types. It must not depend on alignment, reconstruction,
datasets, or CLI modules.

## Invariants

- Public imports go through `tomojax.geometry`, not private `_axes` or `_fov`
  modules.
- Axis helpers preserve NumPy/JAX array type where practical.
- FOV helpers keep the centered-origin grid convention used by existing
  reconstruction tests.
- Gauge canonicalisation transfers mean residual pose components into setup
  parameters while preserving realised setup-plus-pose channels.
- Geometry JSON payloads include a `schema_version`, acquisition metadata, and
  setup parameter metadata. Pose arrays are carried by CSV artifacts.

## Tests

Covered by `tests/test_axes_io.py`, `tests/test_regression_geometry_io.py`,
`tests/test_issue_fix_pr.py`, `tests/test_geometry_gauges.py`,
`tests/test_geometry_serialization.py`, and CLI geometry-build tests.
