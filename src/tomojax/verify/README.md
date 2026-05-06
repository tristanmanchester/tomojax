# tomojax.verify

## Purpose

`tomojax.verify` owns run verification, artifact indexing, recovery reports,
gauge reports, backend provenance reports, and failure classification.

## Public API

- `ArtifactValidationIssue`
- `ArtifactValidationReport`
- `ArtifactValidationError`
- `inspect_run_artifacts(run_dir)`
- `validate_run_artifacts(run_dir)`

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
- Required run artifacts must fail loudly when missing or malformed.

## Tests

- `tests/test_v2_module_skeleton.py` verifies this facade exists and imports.
- `tests/test_verify_artifacts.py` validates positive and negative artifact
  bundle checks.
