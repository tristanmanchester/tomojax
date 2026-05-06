# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 verification semantics
- Goal: prevent rejected Schur geometry updates from counting as verified levels.

### Scope

- In scope:
  - Require accepted Schur diagnostics for level verification when a geometry
    update is executed.
  - Cover the stopped-reconstruction setup-global sidecar case where
    reconstruction absorbs the residual and Schur rejects the update.
  - Record the changed verification semantics and focused validation.
- Out of scope:
  - Adding or changing artifact/report/observability fields.
  - New benchmark ingestion behavior.
  - Solver tuning beyond command-line flags already supported.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align` alternating orchestration and verification.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add Schur-accepted requirement to executed geometry-update verification.
- [x] Add focused stopped-reconstruction sidecar regression.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the verification semantics slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_orchestration.py tests/test_alternating_solver_smoke.py`
  passed: 3 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_orchestration.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_orchestration.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py -q`
  passed: 11 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- A level with a rejected Schur update is not verified, even if the stopped
  reconstruction makes projection loss non-increasing.

### Risks

- Risk: this may make existing smoke summaries stricter.
- Mitigation: cover both the rejected sidecar case and existing alternating
  smoke behavior in focused tests.
