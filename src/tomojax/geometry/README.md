# tomojax.geometry

`tomojax.geometry` provides geometry metadata, setup/pose state, gauge
canonicalisation, detector/axis calibration primitives, and field-of-view
helpers.

## Public API

- Axis order constants and helpers.
- Concrete geometry metadata: `Grid`, `Detector`, `Geometry`,
  `ParallelGeometry`, `LaminographyGeometry`, `RotationAxisGeometry`, and the
  cone-beam `ConeBeam`, `ConeGeometry` and `ConeSegments`.
- FOV helpers such as `compute_roi`, `grid_from_detector_fov`, and
  `cylindrical_mask_xy`.
- State types: `ScalarParameter`, `SetupParameters`, `PoseParameters`,
  `AcquisitionParameters`, and `GeometryState`.
- Calibration/gauge helpers for detector grids, axis state, and calibrated
  metadata patches.
- JSON/CSV artifact helpers for geometry and pose state.

## Dependency policy

Import the geometry classes and field-of-view helpers from `tomojax.geometry`,
and the state, calibration, axis and artifact helpers from
`tomojax.geometry.api`, not from private implementation files.
