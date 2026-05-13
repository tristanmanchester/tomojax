# tomojax.calibration

## Status

`tomojax.calibration` is a provisional internal support package for geometry
calibration value types and low-level detector/axis helpers. It is not a
standalone user workflow.

## Current Public Surface

`tomojax.calibration.api` and the package root intentionally export only
schema/value types:

- `CalibrationState`
- `CalibrationVariable`
- `DetectorPixelScale`
- `DetectorPixelValue`

## Boundary Rule

Calibration estimation should remain owned by `tomojax.align` until the v2 plan
promotes calibration into a first-class deep module. New production workflows
should avoid importing calibration helper internals directly unless they are the
module owner for geometry application or artifact serialization.
