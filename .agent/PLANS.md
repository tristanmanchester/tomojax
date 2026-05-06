# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 nuisance models and weak DOF handling
- Goal: add setup-correlation evidence to weak-DOF policy reports.

### Scope

- In scope:
  - Record det_v setup-correlation evidence from Schur diagnostics.
  - Gate det_v report-only policy on curvature, accepted step, validation
    improvement, and correlation evidence.
  - Preserve theta-scale frozen handling.
- Out of scope:
  - New nuisance solver blocks.
  - Automatic weak-DOF activation changes.
  - Broader failure-classifier policy changes.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.align` with public `tomojax.nuisance` payloads.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add det_v correlation evidence payload.
- [x] Gate det_v report-only decision on correlation evidence.
- [x] Add focused observability artifact tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the weak-DOF correlation slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py -q`
  passed: 10 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Correlation evidence is report-only and does not yet change active parameter
  sets during solve.

### Risks

- Risk: setup-correlation evidence can be unavailable if no Schur diagnostics
  are present.
- Mitigation: keep missing evidence explicit and conservative.
