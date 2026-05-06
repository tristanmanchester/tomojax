# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 nuisance models and weak DOF handling
- Goal: make failure reports surface unmodelled nuisance warnings.

### Scope

- In scope:
  - Set failure-report status to warning when warning-class gates fail.
  - Preserve passing status for clean smoke runs.
  - Add focused classification coverage for structured nuisance residuals.
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

- [x] Set warning status when warning gates fail.
- [x] Add structured residual nuisance classification test.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the failure-report warning slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_verification.py tests/test_failure_report_classification.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating_verification.py tests/test_failure_report_classification.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_verification.py tests/test_failure_report_classification.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_failure_report_classification.py tests/test_alternating_solver_smoke.py -q`
  passed: 11 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Warning status is driven only by warning-class gates in this slice.

### Risks

- Risk: existing clean smoke artifacts should remain status `passed`.
- Mitigation: keep the clean-smoke assertion and add a structured residual
  warning-only case.
