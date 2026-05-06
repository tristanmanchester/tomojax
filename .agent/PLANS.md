# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Milestone 0 cleanup — legacy Ruff unblock
- Goal: clear the first alignment model lint blockers in diagnostics and DOF
  name resolution.

### Scope

- In scope:
  - Add missing diagnostics and DOF module docstrings.
  - Move annotation-only imports behind `TYPE_CHECKING`.
  - Add missing public method/function docstrings in touched modules.
  - Clean local diagnostics simplification lint.
  - Run focused Ruff checks and alignment model tests.
- Out of scope:
  - Alignment algorithm changes.
  - Remaining model package files outside diagnostics/dofs.
  - Repository-wide legacy Ruff cleanup outside the touched model files.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Clean diagnostics lint.
- [x] Clean DOF-name resolution lint.
- [x] Run focused validation.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the cleanup slice if validations pass.

### Validation

- `uv run ruff format src/tomojax/align/model/diagnostics.py src/tomojax/align/model/dofs.py`
  passed.
- `uv run ruff check src/tomojax/align/model/diagnostics.py src/tomojax/align/model/dofs.py`
  passed.
- `uv run pytest tests/test_alignment_gauge_registry.py tests/test_align_quick.py tests/test_align_profiles.py -q`
  passed: 34 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. `diagnostics.py` and `dofs.py` are no longer in the failure list;
  the first remaining blockers start in `src/tomojax/align/model/dof_specs.py`,
  followed by gauge, motion model, schedule, state, and broader repository lint
  backlog. Formatter churn from `just check` was reverted outside this slice.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Kept gauge-policy and DOF-resolution behavior unchanged; changes are limited
  to public docstrings, annotation-only imports, and a local conditional
  simplification.
- Deviation: none from the cleanup scope.

### Risks

- Risk: annotation-only import movement can hide a runtime dependency.
- Mitigation: move only names used in annotations and run focused model/profile
  tests.
- Proposed next fix for `just check`: continue through `dof_specs.py`, `gauge.py`,
  `motion_models.py`, `schedules.py`, and `state.py`.
