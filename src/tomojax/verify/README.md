# tomojax.verify

## Purpose

`tomojax.verify` owns run verification, artifact indexing, recovery reports,
gauge reports, backend provenance reports, and failure classification.

## Public API

- `ArtifactValidationIssue`
- `ArtifactValidationReport`
- `ArtifactValidationError`
- `FAILURE_CLASSES`
- `failure_report_from_gates(gates)`
- `failure_warnings_from_gates(gates)`
- `inspect_run_artifacts(run_dir)`
- `joint_schur_normal_eq_summary(result)`
- `residual_structure_summary(residual, mask)`
- `validate_run_artifacts(run_dir)`
- `write_joint_schur_normal_eq_summary(result, path)`

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
- Residual-structure checks are warning-oriented classifiers, not solver state
  mutations.
- Schur normal-equation summaries are report-only views over solver results and
  must not mutate alignment state.

## Tests

- `tests/test_v2_module_skeleton.py` verifies this facade exists and imports.
- `tests/test_verify_artifacts.py` validates positive and negative artifact
  bundle checks.
- `tests/test_alternating_solver_smoke.py` covers the residual-structure summary
  used by failure reports.
- `tests/test_joint_schur_lm.py` covers Schur normal-equation summary reports.
