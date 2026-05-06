# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Milestone 0 cleanup — legacy Ruff unblock
- Goal: remove `align_multires` orchestration complexity with private
  setup/loop/finalization extraction.

### Scope

- In scope:
  - Extract multires input setup and validation.
  - Extract resume/progress initialization.
  - Extract per-level orchestration from the public function.
  - Preserve checkpoint/resume/observer behavior.
  - Run focused Ruff checks and multires/checkpoint tests.
- Out of scope:
  - Alignment algorithm changes.
  - Geometry module lint cleanup.
  - Repository-wide legacy Ruff cleanup outside this function.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Extract multires input setup.
- [x] Extract resume/progress initialization.
- [x] Extract level orchestration helper.
- [x] Run focused validation.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the cleanup slice if validations pass.

### Validation

- `uv run ruff format src/tomojax/align/_stage_loop.py` passed.
- `uv run ruff check src/tomojax/align/_stage_loop.py` passed.
- `uv run pytest tests/test_multires.py tests/test_bilevel_setup_alignment.py tests/test_align_checkpoint.py tests/test_align_quick.py -q`
  passed: 66 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. `_stage_loop.py` is no longer in the failure list; the first
  remaining blockers are geometry module doc/import lint findings, followed by
  checkpoint/io/model lint and broader repository backlog. Formatter churn from
  `just check` was reverted outside this slice.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Added private multires context and run-state carriers for setup, resume, and
  finalization state.
- Moved translation seeding into a private helper while preserving the existing
  coarsest-level phase-correlation behavior.
- Moved per-level execution and completion checkpoint emission behind a private
  level runner.
- Deviation: none from the cleanup scope.

### Risks

- Risk: moving the per-level loop could alter checkpoint/resume state
  bookkeeping.
- Mitigation: keep carriers private, copy the same mutable state transitions,
  and run focused multires/checkpoint tests.
- Proposed next fix for `just check`: geometry module doc/import cleanup after
  `_stage_loop.py` complexity is gone.
