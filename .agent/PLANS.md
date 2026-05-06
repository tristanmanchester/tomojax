# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Milestone 0 cleanup — legacy Ruff unblock
- Goal: clear the first geometry deep-module lint blockers after `_stage_loop.py`
  cleanup.

### Scope

- In scope:
  - Add missing module/class/function docstrings in
    `src/tomojax/align/geometry`.
  - Replace parent-relative imports with absolute imports in touched files.
  - Move annotation-only imports behind `TYPE_CHECKING`.
  - Fix small local Ruff blockers in parametrization helpers.
  - Run focused Ruff checks and geometry tests.
- Out of scope:
  - Alignment algorithm changes.
  - Checkpoint/io/model lint cleanup outside the geometry package.
  - Repository-wide legacy Ruff cleanup outside this function.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Clean geometry import/type-checking lint.
- [x] Add missing public docstrings in touched geometry modules.
- [x] Fix small parametrization Ruff findings.
- [x] Run focused validation.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the cleanup slice if validations pass.

### Validation

- `uv run ruff format src/tomojax/align/geometry/detector_center.py src/tomojax/align/geometry/geometry_applier.py src/tomojax/align/geometry/geometry_blocks.py src/tomojax/align/geometry/initializers.py src/tomojax/align/geometry/parametrizations.py`
  passed.
- `uv run ruff check src/tomojax/align/geometry/detector_center.py src/tomojax/align/geometry/geometry_applier.py src/tomojax/align/geometry/geometry_blocks.py src/tomojax/align/geometry/initializers.py src/tomojax/align/geometry/parametrizations.py`
  passed.
- `uv run pytest tests/test_geometry.py tests/test_geometry_applier.py tests/test_geometry_block_taxonomy_generator.py tests/test_detector_center_objective.py tests/test_align_quick.py -q`
  passed: 54 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. The touched geometry files are no longer in the failure list; the
  first remaining blockers are `align/io/checkpoint.py`,
  `align/io/params_export.py`, and `align/model/*`, followed by broader
  repository lint backlog. Formatter churn from `just check` was reverted
  outside this slice.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Kept geometry helper exports unchanged and limited the slice to docstrings,
  annotation-only import movement, absolute imports, and direct attribute reads.
- Preserved the existing pose composition order while replacing non-ASCII
  comment text and the unnecessary transform-return assignment.
- Deviation: none from the cleanup scope.

### Risks

- Risk: moving annotation-only imports can accidentally hide runtime
  dependencies.
- Mitigation: only move names used in annotations and run focused geometry tests.
- Proposed next fix for `just check`: checkpoint/io/model lint cleanup after
  the geometry package blockers are clear.
