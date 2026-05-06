# tomojax.cli

## Purpose

`tomojax.cli` owns command-line entrypoints and user-facing orchestration. In
the v2 architecture it should expose simple product commands while hiding
method-level solver choices behind profiles.

The package still contains transitional pre-v2 command modules. This README and
facade mark the intended deep-module boundary for future migration.

## Public API

No Python public API is exported yet. Command modules remain importable for the
existing CLI tests and scripts.

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
- User-facing alignment modes should stay high-level (`off`, `pose`, `auto`,
  `max`) rather than exposing grid-search internals as the default product.

## Tests

- Existing CLI tests cover transitional behavior.
- `tests/test_v2_module_skeleton.py` verifies the v2 facade exists and imports.
