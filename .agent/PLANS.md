# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 7 — alternating solver and continuation
- Goal: wire residual-filter continuation into the Phase 7 smoke geometry loss.

### Scope

- In scope:
  - Add residual-filter schedules to continuation levels.
  - Apply those filters to the projection-domain geometry loss used by the
    alternating smoke runner.
  - Record residual-filter kinds in summaries/artifacts and tests.
- Out of scope:
  - Further legacy Ruff cleanup.
  - GPU/Pallas fast paths.
  - Full production dataset loading through the new command.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Add residual-filter configs to continuation schedules.
- [x] Apply filters in alternating smoke projection loss.
- [x] Record and test residual-filter metadata.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the Phase 7 residual-filter slice.

### Validation

- `uv run ruff format src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py tests/test_continuation_schedules.py`
  passed.
- `uv run ruff check src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py tests/test_continuation_schedules.py`
  passed.
- `uv run basedpyright src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py tests/test_continuation_schedules.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_continuation_schedules.py -q`
  passed: 9 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Use the public `tomojax.forward` residual-filter API from `tomojax.align`.
- Apply filters only to the geometry verification/update loss in this slice;
  reconstruction FISTA remains the existing reference objective.

### Risks

- Risk: filtered loss changes smoke loss values.
- Mitigation: keep assertions structural/deterministic and verify coarse exit
  behavior still holds.
