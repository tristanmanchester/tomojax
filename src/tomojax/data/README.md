# tomojax.data

## Purpose

`tomojax.data` contains lower-level persistence and generator internals used by
the v2 IO, dataset, and simulation modules. It is not the production entrypoint
for user data loading.

This module is transitional and developer/evidence-facing. New production code
should prefer `tomojax.io` for measured data and `tomojax.datasets` for
deterministic synthetic generation.

## Public API

`tomojax.data.api` and the package root re-export the retained lower-level
surface for tests and migration-only code:

- persistence and validation types:
  `LoadedNXTomo`, `NXTomoMetadata`, `ValidationReport`
- simulation types:
  `SimConfig`, `SimulatedData`, `SimulationArtefacts`
- NXtomo/HDF5 functions:
  `load_nxtomo`, `save_nxtomo`, `validate_nxtomo`
- simulation functions:
  `apply_simulation_artefacts`, `simulate`, `simulate_to_file`
- retained phantom helpers:
  `blobs`, `cube`, `lamino_disk`, `random_cubes_spheres`,
  `rotated_centered_cube`, `shepp_logan_3d`, `sphere`

This is not the preferred production import path for new features.
The historical `lamino_disk_legacy` shim remains in
`tomojax.data.phantoms` only for regression coverage and is intentionally not
re-exported from package facades.

## Owned Concepts

This package still owns lower-level NXtomo/HDF5 persistence, retained phantom
helpers, raw preprocessing internals retained behind the IO facade, and
simulation helpers used by existing tests and developer commands.

Dark/flat correction math now lives behind the `tomojax.io` public facade:

- `flat_dark_to_transmission(...)`
- `flat_dark_to_absorption(...)`
- `transmission_to_absorption(...)`
- `absorption_to_transmission(...)`

`tomojax.data.contrast` remains only as a compatibility shim for retained
migration callers and is not re-exported from the package root.

Production file-format loading, provenance, inspection, quicklooks, and CLI
preprocessing are exposed through `tomojax.io`, not by importing this package
directly.

## Intended Migration

- Real measured datasets should enter through `tomojax.io`.
- Deterministic synthetic generation and phantom workflows should live in
  `tomojax.datasets`.
- Public reconstruction and alignment commands should not import this package
  directly.

## Allowed Dependencies

`tomojax.data` should remain low in the dependency graph. It may import:

- `tomojax.core`

It may also use third-party numerical, HDF5, and imaging libraries needed for
retained persistence, phantom, preprocessing, and simulation internals.

## Forbidden Dependencies

New production code should not add dependencies on `tomojax.data`. Wrap required
behavior in the owning v2 deep module first, then migrate call sites.

`tomojax.data` must not import higher-level production modules such as
`tomojax.align`, `tomojax.bench`, `tomojax.cli`, `tomojax.datasets`,
`tomojax.forward`, `tomojax.geometry`, `tomojax.io`, `tomojax.recon`, or
`tomojax.verify`. Production modules must not add direct dependencies on
`tomojax.data`; the import-linter `production-no-data` contract enforces the
current boundary.

## Numerical Invariants

- Dark/flat correction must clamp denominators with the documented epsilon and
  preserve finite transmission/absorption outputs for finite inputs.
- Absorption/transmission conversion helpers must remain inverse-consistent
  within floating-point tolerance for positive transmissions.
- NXtomo load/save helpers must preserve required array shapes, metadata, and
  validation failure reporting.
- Simulation helpers must remain deterministic from their declared seeds and
  configuration objects.
- Phantom helpers must preserve axis ordering and geometry assumptions expected
  by the dataset and benchmark tests that still exercise them.

## Artifact/Provenance Responsibilities

- retained NXtomo/HDF5 files and validation reports
- simulation artifacts attached by `SimulationArtefacts`
- preprocessing/inspection outputs only when called through the retaining
  developer or IO-facing adapters

Production provenance for user-facing loading, inspection, quicklooks, and CLI
preprocessing is owned by `tomojax.io`.

## Testing Strategy

- IO and NXtomo tests cover retained HDF5 load/save/validation behavior through
  the current facade paths.
- Synthetic dataset and simulation tests cover deterministic generation,
  phantom compatibility, and artifact application while migration continues.
- Contrast-correction changes should include focused numerical tests for finite
  outputs, clamping behavior, and absorption/transmission round trips.
- Import-boundary tests and import-linter contracts must continue to prevent new
  production dependencies on this package.
