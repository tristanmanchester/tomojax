# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 nuisance models and weak DOF handling
- Goal: verify fitted nuisance estimates are recorded in run artifacts.

### Scope

- In scope:
  - Add alternating smoke artifact coverage for gain/offset diagnostics.
  - Add alternating smoke artifact coverage for background diagnostics.
  - Preserve existing artifact schema and solver behavior.
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

- [x] Add artifact-level gain/offset nuisance coverage.
- [x] Add artifact-level background nuisance coverage.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the nuisance artifact coverage slice.

### Validation

- `uv run ruff format tests/test_alternating_solver_smoke.py` passed.
- `uv run ruff check tests/test_alternating_solver_smoke.py` passed.
- `uv run basedpyright tests/test_alternating_solver_smoke.py` passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py -q`
  passed: 10 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- This slice adds coverage only; the previous solver diagnostics payload is the
  implementation under test.

### Risks

- Risk: artifact coverage could add runtime by running more smoke profiles.
- Mitigation: reuse existing lightning nuisance smoke tests.
