# tomojax.calibration

## Purpose

`tomojax.calibration` is a provisional internal support package for geometry
calibration value types and low-level detector/axis helpers. It is not a
standalone user workflow.

This module is transitional and developer/evidence-facing. Production workflows
should import calibration-facing detector and axis concepts through
`tomojax.geometry` unless the v2 plan promotes calibration into a first-class
production deep module.

## Public API

`tomojax.calibration.api` and the package root intentionally export only
schema/value types:

- `CalibrationState`
- `CalibrationVariable`
- `DetectorPixelScale`
- `DetectorPixelValue`

## Owned Concepts

- provisional calibration state and variable containers
- detector pixel scale/value wrappers
- low-level detector-grid and axis-geometry helpers
- calibration-unit and convention helpers used while the production boundary is
  still owned by geometry/alignment
- internal JSON serialization helpers for calibration metadata

## Allowed Dependencies

`tomojax.calibration` should remain close to the bottom of the dependency graph.
It may import:

- `tomojax.core`

It may also use standard-library typing/dataclass/serialization facilities and
third-party numerical primitives needed by value-type implementations.

## Forbidden Dependencies

Calibration estimation should remain owned by `tomojax.align` until the v2 plan
promotes calibration into a first-class deep module. New production workflows
must import detector-grid, axis-direction, calibration state, and calibration
metadata helpers through `tomojax.geometry`. Direct imports from
`tomojax.calibration` are reserved for this package's own tests and
implementation.

`tomojax.calibration` must not import higher-level modules such as
`tomojax.align`, `tomojax.bench`, `tomojax.cli`, `tomojax.datasets`,
`tomojax.forward`, `tomojax.geometry`, `tomojax.io`, `tomojax.recon`, or
`tomojax.verify`. Production modules outside geometry must not depend directly
on this package; the import-linter `production-no-calibration` contract enforces
that boundary.

## Numerical Invariants

- Detector pixel scales and values must preserve units explicitly and reject or
  surface invalid numeric states at construction/validation boundaries.
- Gauge helpers must keep canonicalization deterministic and must not change the
  realized geometry semantics when converting equivalent representations.
- Axis and detector-grid helpers must preserve handedness, axis ordering, and
  pixel-center conventions expected by `tomojax.geometry`.
- Manifest/objective helpers must keep calibration metadata JSON-safe and
  reproducible from the stored value types.

## Artifact/Provenance Responsibilities

- calibration state and variable metadata serialized through internal JSON
  helpers
- detector-grid, axis-direction, gauge, objective, and manifest values consumed
  by geometry/alignment evidence paths

User-facing geometry provenance is owned by `tomojax.geometry` and
`tomojax.align`; this module supplies retained value-type internals only.

## Testing Strategy

- Calibration unit tests cover state, units, detector grid, axis geometry,
  conventions/objectives, and gauge behavior.
- Serialization and manifest changes should include JSON-safety and
  reproducibility checks.
- Geometry-facing changes should add or update tests at the `tomojax.geometry`
  facade, not create a new production dependency on this package.
- Import-boundary tests and import-linter contracts must continue to prevent
  direct production dependencies on `tomojax.calibration`.
