# tomojax.cli

## Purpose

`tomojax.cli` owns command-line entrypoints and user-facing orchestration. The
production-facing surface is the grouped `tomojax` dispatcher, which exposes
simple product commands while hiding method-level solver choices behind
profiles.

The package still contains transitional pre-v2 command modules. They remain
importable for tests and the grouped dispatcher, but the package installs only
the `tomojax` console script.

## Public API

- `tomojax.cli.main`

## Dependencies

Allowed dependencies:

- `tomojax.core`
- public facades from v2 numerical modules

Forbidden dependencies:

- private implementation files from other deep modules
- generic utility modules
- new compatibility aliases for old command paths

## Invariants

- Public commands should emit artifact/provenance paths.
- The top-level `tomojax` command exposes user workflows; diagnostic and
  benchmark probes should not be promoted there by default.
- User-facing alignment modes should stay high-level (`off`, `pose`, `auto`,
  `max`) rather than exposing grid-search internals as the default product.

## Tests

- Existing CLI tests cover transitional behavior.
- `tomojax inspect`, `tomojax validate`, `tomojax ingest`,
  `tomojax preprocess`, `tomojax recon`, `tomojax align`, and
  `tomojax simulate` cover the public workflow.
- `tomojax dev ...` owns benchmark and diagnostic probes that should stay out
  of the product-facing command list.
- `tomojax dev align-auto` is a transitional staged synthetic tomography
  benchmark runner, not the final public auto-alignment interface.
- `tomojax dev misalign` and `tomojax dev loss-bench` are developer
  diagnostics. They route real-data containers through `tomojax.io`, but they
  should not be documented as production reconstruction workflows.
- `tests/test_v2_module_skeleton.py` verifies the v2 facade exists and imports.
