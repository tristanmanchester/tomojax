# tomojax.data

## Status

`tomojax.data` is a transitional package retained while the v2 IO, dataset, and
simulation deep modules absorb its responsibilities. It is not the production
entrypoint for user data loading.

## Intended Migration

- Real measured datasets should enter through `tomojax.io`.
- Deterministic synthetic generation and phantom workflows should live in
  `tomojax.datasets`.
- Public reconstruction and alignment commands should not import this package
  directly.

## Current Responsibilities

This package still owns lower-level NXtomo/HDF5 persistence, legacy phantom
helpers, raw preprocessing internals, and simulation helpers used by existing
tests and transitional commands.

## Boundary Rule

New production code should not add dependencies on `tomojax.data`. Wrap required
behavior in the owning v2 deep module first, then migrate call sites.

## Public API

`tomojax.data.api` and the package root re-export the retained transitional
surface for older tests and migration-only code. This is not the preferred
production import path for new features.
