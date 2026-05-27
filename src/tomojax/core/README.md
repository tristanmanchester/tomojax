# tomojax.core

## Purpose

`tomojax.core` provides shared runtime primitives: logging, progress reporting,
and formatting helpers.

## Public API

- `setup_logging(level="INFO")`
- `log_jax_env()`
- `progress_iter(iterable, total=None, desc="")`
- `format_duration(seconds)`

## Dependencies

Standard library and optional JAX/tqdm only. Must not depend on alignment,
reconstruction, datasets, backends, or CLI.

## Invariants

- Public imports go through `tomojax.core`, not private `_logging`.
- Progress reporting is opt-in via `TOMOJAX_PROGRESS`.
- Missing optional dependencies must not prevent execution.

## Tests

Covered by `tests/test_logging.py` and small module coverage tests.
