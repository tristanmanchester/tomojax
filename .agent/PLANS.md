# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Synthetic benchmark foundation / Phase 7 smoke artifacts
- Goal: persist smoke input arrays and recovery tolerances with the Phase 7
  deterministic run.

### Scope

- In scope:
  - Persist observed projections and mask arrays from the deterministic smoke
    run.
  - Emit a smoke `recovery_tolerances.json` artifact.
  - Add those artifacts to `artifact_index.json` and focused tests.
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

- [x] Write projection and mask input arrays.
- [x] Emit recovery tolerances.
- [x] Extend artifact index/test coverage.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the smoke input artifact slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 6 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Use `.npy` arrays for smoke input artifacts, matching existing synthetic
  dataset artifacts.
- Keep recovery tolerances explicit and smoke-specific.

### Risks

- Risk: the Phase 7 smoke run is not yet a full 128^3 synthetic dataset.
- Mitigation: persist the same core input artifact classes so the smoke run is
  benchmark-auditable.
