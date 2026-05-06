# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 7 — alternating solver and continuation
- Goal: expose the Phase 7 `AlternatingAlignmentSolver` orchestration entrypoint.

### Scope

- In scope:
  - Add a small `AlternatingAlignmentSolver` class owned by
    `tomojax.align`.
  - Route `run_alternating_solver_smoke` through the solver class.
  - Export the solver from `tomojax.align.api` and document it in the align
    README.
  - Add focused tests that the class writes the same deterministic smoke
    artifacts.
- Out of scope:
  - Further legacy Ruff cleanup.
  - CLI integration.
  - GPU/Pallas fast paths.
  - A full general dataset solver interface beyond the deterministic smoke
    vertical slice.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Add `AlternatingAlignmentSolver`.
- [x] Route the smoke function through the solver.
- [x] Export and document the solver.
- [x] Extend focused tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the Phase 7 solver entrypoint slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/align/api.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/align/api.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_continuation_schedules.py -q`
  passed: 9 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- The solver class should be a thin orchestration object over the existing
  deterministic smoke runner, not a parallel code path.
- Keep `tomojax.align.__all__` unchanged; expose the new solver from
  `tomojax.align.api`.

### Risks

- Risk: the solver class is still smoke-profile-only.
- Mitigation: name the method `run_smoke` and record the general dataset
  interface as follow-up work.
