# tomojax.geometry

`tomojax.geometry` provides geometry metadata, setup/pose state, gauge
canonicalisation, detector/axis calibration primitives, and field-of-view
helpers.

## Public API

- Axis order constants and helpers.
- Concrete geometry metadata: `Grid`, `Detector`, `Geometry`,
  `ParallelGeometry`, `LaminographyGeometry`, and `RotationAxisGeometry`.
- FOV helpers such as `compute_roi`, `grid_from_detector_fov`, and
  `cylindrical_mask_xy`.
- State types: `ScalarParameter`, `SetupParameters`, `PoseParameters`,
  `AcquisitionParameters`, and `GeometryState`.
- Calibration/gauge helpers for detector grids, axis state, and calibrated
  metadata patches.
- JSON/CSV artifact helpers for geometry and pose state.

## Dependency policy

Import through `tomojax.geometry`, not private implementation files.
