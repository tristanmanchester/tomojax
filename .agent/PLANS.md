# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Milestone 0 cleanup — legacy Ruff unblock
- Goal: clear alignment motion-model lint blockers.

### Scope

- In scope:
  - Add missing `motion_models.py` module and property docstrings.
  - Move annotation-only imports behind `TYPE_CHECKING`.
  - Run focused Ruff checks and motion-model tests.
- Out of scope:
  - Alignment algorithm changes.
  - Remaining model package files outside `motion_models.py`.
  - Repository-wide legacy Ruff cleanup outside this file.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Clean `motion_models.py` doc/import lint.
- [x] Run focused validation.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the cleanup slice if validations pass.

### Validation

- `uv run ruff format src/tomojax/align/model/motion_models.py` passed.
- `uv run ruff check src/tomojax/align/model/motion_models.py` passed.
- `uv run pytest tests/test_align_motion_models.py -q` passed: 6 tests.
- `just imports` passed.
- `uv run pytest tests/test_align_motion_models.py tests/test_align_chunking.py tests/test_align_optimizers.py -q`
  aborted with a JAX/Optax L-BFGS segmentation fault in
  `tests/test_align_chunking.py::test_align_smooth_pose_model_clips_active_bounds_only`;
  this is the known native L-BFGS path and is not caused by this docs/import
  cleanup. No unrelated formatter churn was present from this validation.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Kept pose model basis, coefficient fitting, and expansion behavior unchanged;
  changes are limited to module/property docstrings, annotation-only import
  movement, `cast` quoting, and sorted `__all__`.
- Deviation: none from the cleanup scope.

### Risks

- Risk: annotation-only import movement can hide a runtime dependency.
- Mitigation: move only names used in annotations and run focused motion-model
  tests.
- Proposed next fix for `just check`: continue through `schedules.py` and
  `state.py`.
