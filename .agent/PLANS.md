# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Milestone 0 cleanup — legacy Ruff unblock
- Goal: clear alignment gauge-fix lint blockers.

### Scope

- In scope:
  - Add missing `gauge.py` module and public typed-dict docstrings.
  - Move annotation-only imports behind `TYPE_CHECKING`.
  - Add missing local loop-body annotations.
  - Run focused Ruff checks and gauge tests.
- Out of scope:
  - Alignment algorithm changes.
  - Remaining model package files outside `gauge.py`.
  - Repository-wide legacy Ruff cleanup outside this file.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Clean `gauge.py` doc/import/annotation lint.
- [x] Run focused validation.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the cleanup slice if validations pass.

### Validation

- `uv run ruff format src/tomojax/align/model/gauge.py` passed.
- `uv run ruff check src/tomojax/align/model/gauge.py` passed.
- `uv run pytest tests/test_align_gauge.py tests/test_alignment_gauge_registry.py tests/test_align_quick.py -q`
  passed: 34 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. `gauge.py` is no longer in the failure list; the first remaining
  blockers start in `src/tomojax/align/model/motion_models.py`, followed by
  schedules, state, objectives, and broader repository lint backlog. Formatter
  churn from `just check` was reverted outside this slice.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Kept gauge-fix behavior unchanged; changes are limited to module/API docs,
  annotation-only imports, and type annotations for the JAX loop body.
- Deviation: none from the cleanup scope.

### Risks

- Risk: annotation-only import movement can hide a runtime dependency.
- Mitigation: move only names used in annotations and run focused gauge tests.
- Proposed next fix for `just check`: continue through `motion_models.py`,
  `schedules.py`, and `state.py`.
