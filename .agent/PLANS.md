# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 nuisance models and weak DOF handling
- Goal: move generic failure-report assembly behind `tomojax.verify`.

### Scope

- In scope:
  - Add a public verify-owned failure-report assembly helper.
  - Use the verify helper from alternating artifact generation.
  - Replace align-private white-box failure warning coverage with public verify
    coverage.
- Out of scope:
  - New nuisance solver blocks.
  - Automatic weak-DOF activation changes.
  - Error-class hard-failure policy changes.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align` with public `tomojax.nuisance` payloads.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add verify-owned failure-report helper.
- [x] Route alternating failure reports through verify helper.
- [x] Replace private white-box test with public verify coverage.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the verify failure-report slice.

### Validation

- `uv run ruff format src/tomojax/verify src/tomojax/align/_alternating_verification.py tests/test_failure_report_classification.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/verify src/tomojax/align/_alternating_verification.py tests/test_failure_report_classification.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/verify src/tomojax/align/_alternating_verification.py tests/test_failure_report_classification.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_failure_report_classification.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py -q`
  passed: 16 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Align still owns gate evidence for now; verify owns the generic report
  envelope, status, classes, and warning payloads.

### Risks

- Risk: moving the report envelope can drift artifact shape.
- Mitigation: keep alternating smoke artifact assertions and public verify unit
  tests.
