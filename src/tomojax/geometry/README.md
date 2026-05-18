# tomojax.geometry

`tomojax.geometry` owns public geometry metadata, setup/pose state, gauge
canonicalisation, calibration-derived detector/axis primitives, and detector
field-of-view helpers shared by datasets, CLI entrypoints, reconstruction, and
alignment.

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

Product code imports geometry through `tomojax.geometry`. Private geometry
implementation files and `tomojax.core.geometry` remain implementation details
except where an owning package deliberately wraps them.
