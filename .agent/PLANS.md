# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Alternating solver and continuation
- Goal: add held-out residual verification to Phase 7 early exit.

### Scope

- In scope:
  - Add a small deterministic held-out view mask for geometry updates.
  - Use training-view masks for Schur geometry updates when enabled.
  - Require held-out residual to pass before coarse early exit.
  - Record held-out residual metrics in level summaries and artifacts.
  - Add focused smoke tests.
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

- [x] Add held-out view mask plumbing.
- [x] Use training mask for Schur updates.
- [x] Add held-out residual pass/fail to early-exit verification.
- [x] Record held-out metrics in summaries/artifacts.
- [x] Add focused smoke tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the held-out residual slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py -q`
  passed: 6 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Hold out the last view in the 4-view smoke profile so Schur trains on three
  views and coarse early exit depends on an unoptimised-view residual check.

### Risks

- Risk: the 32^3 held-out residual is very small and may only prove tolerance
  stability, not broad generalisation.
- Mitigation: record the exact held-out residual before/after values and keep
  this as a deterministic early-exit gate until larger benchmark ingestion is
  available.
