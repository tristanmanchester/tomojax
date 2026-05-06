# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Milestone 0 cleanup — legacy Ruff unblock
- Goal: remove `_run_multires_level_stages` complexity with
  behavior-preserving stage-handler extraction.

### Scope

- In scope:
  - Extract proposal-stage execution.
  - Extract setup-geometry stage execution.
  - Extract pose-alignment stage execution.
  - Preserve level stats/loss/checkpoint/observer behavior.
  - Run focused Ruff checks and multires/setup tests.
- Out of scope:
  - Alignment algorithm changes.
  - Large `align_multires` decomposition.
  - Repository-wide legacy Ruff cleanup outside this function.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Extract proposal stage handler.
- [x] Extract setup stage handler.
- [x] Extract pose stage handler.
- [x] Run focused validation.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the cleanup slice if validations pass.

### Validation

- `uv run ruff format src/tomojax/align/_stage_loop.py` passed.
- `uv run ruff check src/tomojax/align/_stage_loop.py` now reports only
  `align_multires` PLR0912/PLR0915.
- `uv run pytest tests/test_multires.py tests/test_bilevel_setup_alignment.py tests/test_align_checkpoint.py -q`
  passed: 43 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. The first remaining blockers are `align_multires`
  PLR0912/PLR0915, followed by geometry module doc/import findings and the
  broader repository lint backlog. Formatter churn from `just check` was
  reverted outside this slice.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Added a private `StageLoopState` carrier so proposal, setup-geometry, and
  pose-alignment handlers can return the same level state without widening the
  public API.
- Kept stage dispatch behavior local to `_stage_loop.py`; no alignment
  algorithm changes were made.
- Deviation: none from the cleanup scope.

### Risks

- Risk: stage handler extraction could change level stats/loss accumulation.
- Mitigation: keep helpers private, return the same mutated state values, and
  run focused multires/setup/checkpoint tests.
- Proposed next fix for `just check`: split `align_multires` orchestration
  complexity.
