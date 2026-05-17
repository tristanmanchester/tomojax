# tomojax.cli

## Purpose

`tomojax.cli` owns command-line entrypoints and user-facing orchestration. The
production-facing surface is the grouped `tomojax` dispatcher, which exposes
simple product commands while hiding method-level solver choices behind
profiles.

The package keeps command implementations behind the grouped dispatcher. The
installed product entrypoint is the `tomojax` console script; implementation
modules are not separate public commands.

## Public API

- `tomojax.cli.main.main`: grouped console-script dispatcher.
- `tomojax.cli.api.CliCommand`: command metadata record.
- `tomojax.cli.api.product_command_names()`: product-facing grouped commands.
- `tomojax.cli.api.developer_command_names()`: developer diagnostic commands
  grouped under `tomojax dev`.

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
- User-facing alignment modes should stay high-level (`cor`, `pose`, `auto`,
  `max`) rather than exposing grid-search internals as the default product.

## Tests

- Existing CLI tests cover command routing, manifests, and product-facing help.
- `tomojax inspect`, `tomojax validate`, `tomojax ingest`,
  `tomojax preprocess`, `tomojax recon`, `tomojax align`, and
  `tomojax simulate` cover the public workflow.
- `tomojax dev ...` owns benchmark and diagnostic probes that should stay out
  of the product-facing command list.
- `tomojax dev align-auto` is a staged synthetic tomography benchmark runner,
  not the public auto-alignment interface.
- `tomojax dev misalign` and `tomojax dev loss-bench` are developer
  diagnostics. They route real-data containers through `tomojax.io`, but they
  should not be documented as production reconstruction workflows.
- `tests/test_v2_module_skeleton.py` verifies the v2 facade exists and imports.
