# tomojax.verify

## Purpose

`tomojax.verify` will own run verification, artifact indexing, recovery
reports, gauge reports, backend provenance reports, and failure classification.

This package is currently a v2 skeleton facade. It intentionally exposes no
public behavior until verification artifacts are implemented.

## Public API

No public names are exported yet.

## Dependencies

Allowed future dependencies:

- `tomojax.core`
- `tomojax.geometry`
- `tomojax.io`

Forbidden dependencies:

- private implementation files from other deep modules
- mutating solver state
- backend-specific kernels

## Invariants

- Verification artifacts must be machine-readable.
- Synthetic recovery checks must be deterministic from a seed.
- Reports must distinguish failures, warnings, and unverified conditions.

## Tests

- `tests/test_v2_module_skeleton.py` verifies this skeleton facade exists and
  imports.
