# tomojax.data

## Status

`tomojax.data` contains lower-level persistence and generator internals used by
the v2 IO, dataset, and simulation modules. It is not the production entrypoint
for user data loading.

## Intended Migration

- Real measured datasets should enter through `tomojax.io`.
- Deterministic synthetic generation and phantom workflows should live in
  `tomojax.datasets`.
- Public reconstruction and alignment commands should not import this package
  directly.

## Current Responsibilities

This package still owns lower-level NXtomo/HDF5 persistence, retained phantom
helpers, contrast-correction math, raw preprocessing internals retained behind
the IO facade, and simulation helpers used by existing tests and developer
commands.

Dark/flat correction math lives in `tomojax.data.contrast`:

- `flat_dark_to_transmission(...)`
- `flat_dark_to_absorption(...)`

Production file-format loading, provenance, inspection, quicklooks, and CLI
preprocessing are exposed through `tomojax.io`, not by importing this package
directly.

## Boundary Rule

New production code should not add dependencies on `tomojax.data`. Wrap required
behavior in the owning v2 deep module first, then migrate call sites.

## Public API

`tomojax.data.api` and the package root re-export the retained internal surface
for tests and migration-only code. This is not the preferred production import
path for new features.
