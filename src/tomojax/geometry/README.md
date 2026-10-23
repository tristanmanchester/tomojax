# tomojax.geometry

`tomojax.geometry` describes a scan: the reconstruction `Grid`, the `Detector`,
the geometry classes `ParallelGeometry`, `LaminographyGeometry`,
`ConeGeometry` (with `ConeBeam`) and `ConeSegments`, and field-of-view helpers
such as `compute_roi`, `grid_from_detector_fov` and `cylindrical_mask_xy`.

`tomojax.geometry.api` holds the expert helpers alignment, IO and the solvers
build on: calibration state (`CalibrationState`, `GeometryState`, `validate_calibration_gauges`),
axis poses and detector grids.

Import from these two modules, not from private implementation files.
