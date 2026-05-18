# tomojax.core

## Purpose

`tomojax.core` owns small runtime primitives that are shared across numerical
modules and command surfaces. The current Milestone 0 public surface is limited
to logging/progress helpers migrated out of the forbidden `tomojax.utils`
namespace.

## Public API

- `setup_logging(level="INFO")`
- `log_jax_env()`
- `progress_iter(iterable, total=None, desc="")`
- `format_duration(seconds)`

## Dependencies

This module may use the Python standard library and optional JAX/tqdm probes. It
must not depend on alignment, reconstruction, datasets, backend policy, or CLI
modules.

## Invariants

- Public imports go through `tomojax.core`, not private `_logging`.
- Progress reporting is opt-in via `TOMOJAX_PROGRESS`.
- Missing optional dependencies must not prevent normal execution.

## Tests

Covered by `tests/test_logging.py` and small module coverage tests.
