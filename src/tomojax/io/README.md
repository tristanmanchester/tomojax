# tomojax.io

## Purpose

`tomojax.io` owns typed public helpers for serialising TomoJAX metadata,
manifests, and future artifact schemas.

## Public API

- `JsonValue`
- `normalize_json(...)`
- `drop_none(...)`

## Dependencies

This module may depend on `tomojax.core` data structures when artifact schemas
need them. It must not depend on alignment, reconstruction, datasets, or CLI
implementation modules.

## Invariants

- Public helpers return strict JSON-compatible values.
- Non-finite floats are converted to strings so callers can use
  `json.dump(..., allow_nan=False)`.
- Optional NumPy and JAX arrays are converted without making either library a
  module boundary dependency for callers.

## Tests

Covered by `tests/test_json_utils.py` and downstream manifest/checkpoint tests.
