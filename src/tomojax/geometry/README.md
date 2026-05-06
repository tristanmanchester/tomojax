# tomojax.geometry

## Purpose

`tomojax.geometry` owns public geometry metadata helpers that are shared by
datasets, CLI entrypoints, reconstruction, and alignment. The current Milestone
0 surface is limited to axis-order normalization and detector field-of-view
helpers migrated out of the forbidden `tomojax.utils` namespace.

## Public API

- Axis order constants: `INTERNAL_VOLUME_AXES`, `DISK_VOLUME_AXES`,
  `VOLUME_AXES_ATTR`
- Axis helpers: `axes_to_perm`, `transpose_volume`, `infer_disk_axes`,
  `is_shape_xyz`, `is_shape_zyx`
- FOV helpers: `RoiInfo`, `compute_roi`, `grid_from_detector_fov`,
  `grid_from_detector_fov_cube`, `grid_from_detector_fov_slices`,
  `cylindrical_mask_xy`

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

## Tests

Covered by `tests/test_axes_io.py`, `tests/test_regression_geometry_io.py`,
`tests/test_issue_fix_pr.py`, and CLI geometry-build tests.
