# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Milestone 0 cleanup — legacy Ruff unblock
- Goal: remove `_reconstruction_stage.py` Ruff blockers with
  behavior-preserving import, annotation, and stat-helper cleanup.

### Scope

- In scope:
  - Replace parent-relative imports with absolute imports.
  - Move annotation-only imports behind `TYPE_CHECKING`.
  - Add return annotations for local reconstruction runner helpers.
  - Extract final reconstruction stat assembly to clear statement count.
  - Run focused Ruff checks and reconstruction/alignment tests.
- Out of scope:
  - Alignment algorithm changes.
  - Reconstruction numerical formula changes.
  - Repository-wide legacy Ruff cleanup outside this function.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Clean `_reconstruction_stage.py` imports.
- [x] Add local runner return annotations.
- [x] Extract final stat assembly.
- [x] Run focused validation.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the cleanup slice if validations pass.

### Validation

- `uv run ruff check src/tomojax/align/_reconstruction_stage.py` passed.
- `uv run ruff format src/tomojax/align/_reconstruction_stage.py` passed.
- `uv run pytest tests/test_align_quick.py tests/test_align_chunking.py -q -k '(gn or gd or smooth_pose_model or pose_model) and not lbfgs'`
  passed: 47 tests, 5 deselected.
- `uv run pytest tests/test_align_profiles.py -q` passed: 6 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; `_reconstruction_stage.py` is no
  longer in the failure list. The first remaining failures are type-checking
  import findings in `_results.py`, then `_setup_stage.py`, `_stage_loop.py`,
  and later modules/tests. Formatter churn from this command was reverted
  outside this slice.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Decision: keep `Mapping` imported at runtime from `collections.abc` because
  reconstruction info still uses `isinstance(info_rec, Mapping)`.
- Decision: extract only the final `OuterStat` assembly and OOM-message
  predicate needed to clear statement count, leaving reconstruction execution
  paths unchanged.
- Deviation: none from the cleanup scope.

### Risks

- Risk: final-stat extraction could change `OuterStat` payload if keys drift.
- Mitigation: move the existing payload construction verbatim into a private
  helper and run focused reconstruction/alignment tests.
- Proposed next fix for `just check`: continue into `_results.py` and
  `_setup_stage.py`.
